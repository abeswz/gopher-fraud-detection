package main

import (
	"log"
	"net"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

const (
	defaultPort    = 9999
	defaultBacklog = 8192
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func createListener(port, backlog int) (int, error) {
	fd, err := unix.Socket(unix.AF_INET, unix.SOCK_STREAM|unix.SOCK_NONBLOCK|unix.SOCK_CLOEXEC, 0)
	if err != nil {
		return -1, err
	}
	unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_REUSEADDR, 1)
	unix.SetsockoptInt(fd, unix.SOL_SOCKET, unix.SO_REUSEPORT, 1)
	unix.SetsockoptInt(fd, unix.IPPROTO_TCP, unix.TCP_DEFER_ACCEPT, 1)

	addr := &unix.SockaddrInet4{Port: port}
	copy(addr.Addr[:], net.ParseIP("0.0.0.0").To4())

	if err := unix.Bind(fd, addr); err != nil {
		unix.Close(fd)
		return -1, err
	}
	if err := unix.Listen(fd, backlog); err != nil {
		unix.Close(fd)
		return -1, err
	}
	return fd, nil
}

func main() {
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(10 << 20)

	unix.Prctl(unix.PR_SET_TIMERSLACK, 1, 0, 0, 0)

	port := defaultPort
	if p, err := strconv.Atoi(os.Getenv("PORT")); err == nil && p > 0 {
		port = p
	}

	var upstreams []string
	for s := range strings.SplitSeq(envOr("UPSTREAMS", ""), ",") {
		if s = strings.TrimSpace(s); s != "" {
			upstreams = append(upstreams, s)
		}
	}
	if len(upstreams) == 0 {
		log.Fatal("UPSTREAMS env var required (comma-separated unix socket paths)")
	}

	listenFd, err := createListener(port, defaultBacklog)
	if err != nil {
		log.Fatalf("create listener: %v", err)
	}

	udsFd, err := unix.Socket(unix.AF_UNIX, unix.SOCK_DGRAM, 0)
	if err != nil {
		log.Fatalf("create UDS socket: %v", err)
	}
	unix.SetsockoptInt(udsFd, unix.SOL_SOCKET, unix.SO_SNDBUF, 16*1024*1024)

	epfd, err := unix.EpollCreate1(0)
	if err != nil {
		log.Fatalf("epoll_create1: %v", err)
	}
	if err := unix.EpollCtl(epfd, unix.EPOLL_CTL_ADD, listenFd,
		&unix.EpollEvent{Events: unix.EPOLLIN, Fd: int32(listenFd)}); err != nil {
		log.Fatalf("epoll_ctl: %v", err)
	}

	events := make([]unix.EpollEvent, 1024)
	batches := make([][]int, len(upstreams))
	for i := range batches {
		batches[i] = make([]int, 0, 64)
	}
	var rr int

	log.Printf("lb started: port=%d upstreams=%v", port, upstreams)

	for {
		n, err := unix.EpollWait(epfd, events, -1)
		if err != nil {
			if err == unix.EINTR {
				continue
			}
			log.Fatalf("epoll_wait: %v", err)
		}

		for i := range batches {
			batches[i] = batches[i][:0]
		}

		for i := 0; i < n; i++ {
			if int(events[i].Fd) != listenFd {
				continue
			}
			for {
				cfd, _, err := unix.Accept4(listenFd, unix.SOCK_NONBLOCK|unix.SOCK_CLOEXEC)
				if err != nil {
					break
				}
				unix.SetsockoptInt(cfd, unix.IPPROTO_TCP, unix.TCP_NODELAY, 1)
				batches[rr%len(upstreams)] = append(batches[rr%len(upstreams)], cfd)
				rr++
			}
		}

		for i, batch := range batches {
			if len(batch) == 0 {
				continue
			}
			addr := &unix.SockaddrUnix{Name: upstreams[i]}
			for j := 0; j < len(batch); j += 16 {
				end := min(j+16, len(batch))
				rights := unix.UnixRights(batch[j:end]...)
				if err := unix.Sendmsg(udsFd, []byte{0}, rights, addr, 0); err != nil {
					log.Printf("sendmsg to %s: %v", upstreams[i], err)
				}
				for _, fd := range batch[j:end] {
					unix.Close(fd)
				}
			}
		}
	}
}
