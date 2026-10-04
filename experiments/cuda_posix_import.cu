// Diagnostic only. No NCCL and no BFS: isolate CUDA VMM POSIX import.
// Run as two independent processes through the existing window-pair launcher.
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <poll.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>
#include <stdexcept>
#include <string>

static void driver(CUresult rc, const char* stage) {
  if (rc == CUDA_SUCCESS) return;
  const char* name = "unknown";
  cuGetErrorName(rc, &name);
  throw std::runtime_error(std::string(stage) + ":" + name);
}
static void runtime(cudaError_t rc, const char* stage) {
  if (rc != cudaSuccess)
    throw std::runtime_error(std::string(stage) + ":" + cudaGetErrorString(rc));
}
static void entering(int rank, const char* stage) {
  std::printf("rank=%d stage=%s result=ENTER\n", rank, stage);
  std::fflush(stdout);
}
struct Fd {
  int value = -1;
  ~Fd() { if (value >= 0) close(value); }
};
static void ready(int fd, short events) {
  pollfd p{fd, events, 0};
  int rc;
  do { rc = poll(&p, 1, 30000); } while (rc < 0 && errno == EINTR);
  if (rc != 1 || !(p.revents & events)) throw std::runtime_error("socket_timeout_or_failure");
}
static void barrier(int fd) {
  char byte = 'B';
  if (send(fd, &byte, 1, MSG_NOSIGNAL) != 1) throw std::runtime_error("barrier_send");
  ready(fd, POLLIN);
  if (recv(fd, &byte, 1, 0) != 1 || byte != 'B') throw std::runtime_error("barrier_recv");
}
static void send_fd(int socket, int handle, size_t bytes) {
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
  iovec io{&bytes, sizeof(bytes)};
  msghdr msg{};
  msg.msg_iov = &io; msg.msg_iovlen = 1;
  msg.msg_control = control; msg.msg_controllen = sizeof(control);
  auto* c = CMSG_FIRSTHDR(&msg);
  c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(c), &handle, sizeof(handle));
  if (sendmsg(socket, &msg, MSG_NOSIGNAL) != sizeof(bytes)) throw std::runtime_error("fd_send");
}
static int receive_fd(int socket, size_t expected) {
  ready(socket, POLLIN);
  alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))]{};
  size_t bytes = 0;
  iovec io{&bytes, sizeof(bytes)};
  msghdr msg{};
  msg.msg_iov = &io; msg.msg_iovlen = 1;
  msg.msg_control = control; msg.msg_controllen = sizeof(control);
  if (recvmsg(socket, &msg, MSG_CMSG_CLOEXEC) != sizeof(bytes) ||
      (msg.msg_flags & (MSG_CTRUNC | MSG_TRUNC))) throw std::runtime_error("fd_recv");
  auto* c = CMSG_FIRSTHDR(&msg);
  if (!c || c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS ||
      c->cmsg_len != CMSG_LEN(sizeof(int))) throw std::runtime_error("fd_control");
  int fd = -1;
  std::memcpy(&fd, CMSG_DATA(c), sizeof(fd));
  if (bytes != expected) { close(fd); throw std::runtime_error("granularity_mismatch"); }
  return fd;
}
struct Mapping {
  CUmemGenericAllocationHandle handle = 0;
  CUdeviceptr address = 0;
  size_t bytes = 0;
  bool mapped = false;
  ~Mapping() {
    if (mapped) cuMemUnmap(address, bytes);
    if (address) cuMemAddressFree(address, bytes);
    if (handle) cuMemRelease(handle);
  }
  void map(int device) {
    driver(cuMemAddressReserve(&address, bytes, 0, 0, 0), "reserve");
    driver(cuMemMap(address, bytes, 0, handle, 0), "map");
    mapped = true;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    driver(cuMemSetAccess(address, bytes, &access, 1), "set_access");
  }
};
__global__ void check_values(const unsigned* own, const unsigned* peer,
                             unsigned* result, unsigned expected) {
  if (!blockIdx.x && !threadIdx.x)
    *result = (own[0] == expected && peer[0] == (expected ^ 1u)) ? 0u : 1u;
}
int main(int argc, char** argv) {
  const bool local = argc == 2 && std::strcmp(argv[1], "local") == 0;
  if (argc != 2 || (!local && std::strcmp(argv[1], "import") != 0)) return 1;
  const char* rank_env = std::getenv("MGBFS_WINDOW_RANK");
  const char* path_env = std::getenv("MGBFS_WINDOW_BOOTSTRAP");
  if (!rank_env || !path_env || (std::strcmp(rank_env,"0") && std::strcmp(rank_env,"1"))) return 1;
  const int rank = rank_env[0] - '0';
  const std::string path = std::string(path_env) + ".socket";
  bool owns_path = false;
  try {
    entering(rank, "context");
    runtime(cudaSetDevice(rank), "set_device");
    runtime(cudaFree(nullptr), "context");
    Fd connection, listener;
    if (!local) {
      sockaddr_un address{};
      address.sun_family = AF_UNIX;
      if (path.size() >= sizeof(address.sun_path)) throw std::runtime_error("socket_path_length");
      std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
      if (rank == 0) {
        listener.value = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (listener.value < 0 || bind(listener.value, reinterpret_cast<sockaddr*>(&address), sizeof(address)))
          throw std::runtime_error("socket_bind");
        owns_path = true;
        if (listen(listener.value, 1)) throw std::runtime_error("socket_listen");
        ready(listener.value, POLLIN);
        connection.value = accept4(listener.value, nullptr, nullptr, SOCK_CLOEXEC);
        if (connection.value < 0) throw std::runtime_error("socket_accept");
      } else {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (true) {
          connection.value = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
          if (connection.value < 0) throw std::runtime_error("socket_create");
          if (!connect(connection.value, reinterpret_cast<sockaddr*>(&address), sizeof(address))) break;
          const int error = errno;
          close(connection.value); connection.value = -1;
          if ((error != ENOENT && error != ECONNREFUSED) || std::chrono::steady_clock::now() >= end)
            throw std::runtime_error("socket_connect");
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
    }
    Mapping own, peer;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = rank;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    driver(cuMemGetAllocationGranularity(&own.bytes, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
    entering(rank, "local_create");
    driver(cuMemCreate(&own.handle, own.bytes, &prop, 0), "create");
    entering(rank, "local_map");
    own.map(rank);
    const unsigned expected = 100u + rank;
    // Initialize before export; diagnostic synchronizations are intentional.
    entering(rank, "initialize");
    driver(cuMemsetD32(own.address, expected, own.bytes / sizeof(unsigned)), "initialize");
    runtime(cudaDeviceSynchronize(), "initialized");
    std::printf("rank=%d stage=vmm_local_initialized result=PASS\n", rank); std::fflush(stdout);
    if (!local) {
      Fd exported, imported;
      entering(rank, "export_exchange");
      driver(cuMemExportToShareableHandle(&exported.value, own.handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), "export");
      send_fd(connection.value, exported.value, own.bytes);
      imported.value = receive_fd(connection.value, own.bytes);
      peer.bytes = own.bytes;
      entering(rank, "peer_import");
      driver(cuMemImportFromShareableHandle(&peer.handle, reinterpret_cast<void*>(static_cast<intptr_t>(imported.value)),
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), "import");
      entering(rank, "peer_map");
      peer.map(rank);
      barrier(connection.value);
    }
    // Result lives in our own initialized VMM allocation, not another allocator.
    auto* own_ptr = reinterpret_cast<unsigned*>(own.address);
    if (local) {
      driver(cuMemsetD32(own.address + sizeof(unsigned), expected ^ 1u, 1), "local_peer_initialize");
    }
    entering(rank, "check_launch");
    check_values<<<1,32>>>(own_ptr, local ? own_ptr + 1 : reinterpret_cast<unsigned*>(peer.address),
                          own_ptr + 2, expected);
    runtime(cudaGetLastError(), "check_launch");
    runtime(cudaDeviceSynchronize(), "check_complete");
    unsigned result = 1;
    driver(cuMemcpyDtoH(&result, own.address + 2 * sizeof(unsigned), sizeof(result)), "result");
    if (result) throw std::runtime_error("value_mismatch");
    if (!local) barrier(connection.value); // No exporter release while the peer reads.
    std::printf("rank=%d stage=vmm_%s result=PASS\n", rank, local ? "local" : "import"); std::fflush(stdout);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "rank=%d stage=vmm_failure error=%s\n", rank, error.what());
    if (owns_path) unlink(path.c_str());
    return 2;
  }
  if (owns_path) unlink(path.c_str());
  return 0;
}
