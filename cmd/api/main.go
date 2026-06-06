package main

import (
	"bytes"
	"log/slog"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"unsafe"

	"gopher-fraud-detection/internal/search"
	"gopher-fraud-detection/internal/service"
	"gopher-fraud-detection/internal/vectorizer"

	"golang.org/x/sys/unix"
)

const (
	bufSize   = 4096
	maxFDs    = 1024
	maxEvents = 128
)

type connState struct {
	buf [bufSize]byte
	pos int
}

var (
	states  []connState
	epollFD int

	hdrSep = []byte("\r\n\r\n")
	clKey  = []byte("content-length:")

	// Pre-rendered responses: count 0..5.
	responses = [6][]byte{
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}"),
		[]byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}"),
	}
	readyResp = []byte("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
	errResp   = []byte("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func sendAll(fd int, p []byte) {
	for len(p) > 0 {
		n, err := unix.Write(fd, p)
		if err == unix.EINTR {
			continue
		}
		if n > 0 {
			p = p[n:]
		}
		if err != nil {
			return
		}
	}
}

func contentLength(hdr []byte) int {
	i := indexFold(hdr, clKey)
	if i < 0 {
		return -1
	}
	j := i + len(clKey)
	for j < len(hdr) && (hdr[j] == ' ' || hdr[j] == '\t') {
		j++
	}
	n := 0
	for j < len(hdr) && hdr[j] >= '0' && hdr[j] <= '9' {
		n = n*10 + int(hdr[j]-'0')
		if n > bufSize {
			return bufSize + 1
		}
		j++
	}
	return n
}

func indexFold(hay, needle []byte) int {
	if len(needle) == 0 {
		return 0
	}
	last := len(hay) - len(needle)
	for i := 0; i <= last; i++ {
		k := 0
		for ; k < len(needle); k++ {
			c := hay[i+k]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != needle[k] {
				break
			}
		}
		if k == len(needle) {
			return i
		}
	}
	return -1
}

func handleRequest(req []byte, bodyOff int) []byte {
	n := len(req)
	if n >= 4 && req[0] == 'G' && req[1] == 'E' && req[2] == 'T' {
		return readyResp
	}
	if n < 5 || req[0] != 'P' || req[1] != 'O' || req[2] != 'S' || req[3] != 'T' {
		return errResp
	}
	body := req[bodyOff:]
	count := service.CalculateFraudScoreRaw(body)
	if count < 0 || count > 5 {
		return errResp
	}
	return responses[count]
}

func closeClient(fd int) {
	unix.EpollCtl(epollFD, unix.EPOLL_CTL_DEL, fd, nil)
	unix.Close(fd)
	if fd < maxFDs {
		states[fd].pos = 0
	}
}

func handleClientEvent(fd int) {
	st := &states[fd]
	if st.pos >= bufSize {
		closeClient(fd)
		return
	}
	n, err := unix.Read(fd, st.buf[st.pos:])
	if err == unix.EAGAIN || err == unix.EWOULDBLOCK || err == unix.EINTR {
		return
	}
	if n <= 0 || err != nil {
		closeClient(fd)
		return
	}
	st.pos += n

	for st.pos > 0 {
		hdrEnd := bytes.Index(st.buf[:st.pos], hdrSep)
		if hdrEnd < 0 {
			return
		}
		bodyOff := hdrEnd + 4
		cl := contentLength(st.buf[:bodyOff])
		if cl < 0 {
			cl = 0
		}
		total := bodyOff + cl
		if total > bufSize {
			closeClient(fd)
			return
		}
		if st.pos < total {
			return
		}
		sendAll(fd, handleRequest(st.buf[:total], bodyOff))
		rem := st.pos - total
		if rem > 0 {
			copy(st.buf[:rem], st.buf[total:st.pos])
		}
		st.pos = rem
	}
}

type schedParam struct{ priority int32 }

func tryRealtimePriority() {
	if os.Getenv("NO_FIFO") != "" {
		return
	}
	p := schedParam{priority: 10}
	unix.Syscall(unix.SYS_SCHED_SETSCHEDULER, 0, 1, uintptr(unsafe.Pointer(&p)))
}

func serverLoop(listenFD int) {
	runtime.LockOSThread()
	tryRealtimePriority()

	events := make([]unix.EpollEvent, maxEvents)
	oob := make([]byte, 256) // enough for 16 fds via SCM_RIGHTS
	var oobBuf [1]byte
	for {
		n, err := unix.EpollWait(epollFD, events, 1)
		if err == unix.EINTR {
			continue
		}
		for i := 0; i < n; i++ {
			fd := int(events[i].Fd)
			if fd == listenFD {
				for {
					_, oobn, _, _, err := unix.Recvmsg(listenFD, oobBuf[:], oob, unix.MSG_DONTWAIT)
					if err != nil {
						break
					}
					scms, err := unix.ParseSocketControlMessage(oob[:oobn])
					if err != nil {
						break
					}
					for _, scm := range scms {
						fds, err := unix.ParseUnixRights(&scm)
						if err != nil {
							continue
						}
						for _, cfd := range fds {
							if cfd >= maxFDs {
								unix.Close(cfd)
								continue
							}
							unix.SetsockoptInt(cfd, unix.SOL_TCP, unix.TCP_QUICKACK, 1)
							states[cfd].pos = 0
							unix.EpollCtl(epollFD, unix.EPOLL_CTL_ADD, cfd,
								&unix.EpollEvent{Events: unix.EPOLLIN | unix.EPOLLRDHUP, Fd: int32(cfd)})
						}
					}
				}
			} else {
				handleClientEvent(fd)
			}
		}
	}
}

func die(msg string) {
	slog.Error(msg)
	os.Exit(1)
}

func main() {
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(160 << 20)

	unix.Prctl(unix.PR_SET_TIMERSLACK, 1, 0, 0, 0)
	unix.Mlockall(unix.MCL_CURRENT | unix.MCL_FUTURE)

	normPath := envOr("NORM_PATH", "resources/normalization.json")
	mccPath := envOr("MCC_PATH", "resources/mcc_risk.json")
	indexDir := envOr("INDEX_DIR", "index")
	sockPath := envOr("SOCK", "")
	if sockPath == "" {
		die("SOCK env var required")
	}

	vec, err := vectorizer.Load(normPath, mccPath)
	if err != nil {
		die("load vectorizer: " + err.Error())
	}
	service.Vec = vec

	loaded := 0
	for i := 0; i < search.NPartitions; i++ {
		path := indexDir + "/index_p" + strconv.Itoa(i) + ".bin"
		if _, err := os.Stat(path); err != nil {
			continue
		}
		ix, err := search.Open(path)
		if err != nil {
			die("open index_p" + strconv.Itoa(i) + ".bin: " + err.Error())
		}
		service.Indices[i] = ix
		loaded++
	}
	if loaded == 0 {
		die("no index files found in " + indexDir + " — run build_index first")
	}
	slog.Info("loaded partitions", "count", loaded)

	listenFD, err := unix.Socket(unix.AF_UNIX, unix.SOCK_DGRAM|unix.SOCK_NONBLOCK|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		die("socket: " + err.Error())
	}
	unix.Unlink(sockPath)
	if err := unix.Bind(listenFD, &unix.SockaddrUnix{Name: sockPath}); err != nil {
		die("bind: " + err.Error())
	}
	unix.Chmod(sockPath, 0o666)

	epollFD, err = unix.EpollCreate1(unix.EPOLL_CLOEXEC)
	if err != nil {
		die("epoll_create1: " + err.Error())
	}
	unix.EpollCtl(epollFD, unix.EPOLL_CTL_ADD, listenFD,
		&unix.EpollEvent{Events: unix.EPOLLIN, Fd: int32(listenFD)})

	states = make([]connState, maxFDs)
	serverLoop(listenFD)
}
