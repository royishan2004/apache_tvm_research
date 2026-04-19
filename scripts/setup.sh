#!/usr/bin/env bash
set -euo pipefail

########################################
# USER CONFIG
########################################
TVM_VERSION="${TVM_VERSION:-v0.21.0}"

WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)/apache_tvm_research}"
RESEARCH_REPO_URL="${RESEARCH_REPO_URL:-https://github.com/royishan2004/apache_tvm_research.git}"

TVM_ROOT="${TVM_ROOT:-$WORKSPACE_DIR/tvm}"
BUILD_DIR="${BUILD_DIR:-$TVM_ROOT/build}"

VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/venv}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLVM_CONFIG="${LLVM_CONFIG:-llvm-config-14}"
GENERATOR="${GENERATOR:-Ninja}"
JOBS="${JOBS:-$(nproc)}"

ENABLE_RPC="${ENABLE_RPC:-ON}"
ENABLE_GRAPH_EXECUTOR="${ENABLE_GRAPH_EXECUTOR:-ON}"
ENABLE_PROFILER="${ENABLE_PROFILER:-ON}"
ENABLE_OPENMP="${ENABLE_OPENMP:-ON}"

EXECUTION_GUIDE_URL="https://github.com/royishan2004/apache_tvm_research?tab=readme-ov-file#execution-guide-what-to-run-where-and-why"

########################################
# LOGGING
########################################
log() {
    echo
    echo "========================================"
    echo "$1"
    echo "========================================"
}

########################################
# INSTALL SYSTEM DEPENDENCIES
########################################
log "Installing system dependencies"

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    llvm-14 llvm-14-dev llvm-14-tools \
    libedit-dev \
    libxml2-dev \
    libz-dev \
    libffi-dev \
    python3-dev \
    python3-setuptools \
    python3-venv

########################################
# ENSURE PIP IS AVAILABLE
########################################
log "Ensuring pip is available"

if $PYTHON_BIN -m pip --version >/dev/null 2>&1; then
    PIP_CMD="$PYTHON_BIN -m pip"

elif command -v pip3 >/dev/null 2>&1; then
    PIP_CMD="pip3"

else
    echo "pip not found. Attempting ensurepip..."
    if $PYTHON_BIN -m ensurepip --upgrade >/dev/null 2>&1; then
        PIP_CMD="$PYTHON_BIN -m pip"
    else
        echo "ensurepip failed. Installing python3-pip..."
        sudo apt install -y python3-pip
        PIP_CMD="$PYTHON_BIN -m pip"
    fi
fi

echo "Using pip command: $PIP_CMD"

########################################
# INSTALL PYTHON BUILD DEPENDENCIES
########################################
log "Installing Python build dependencies"

$PIP_CMD install --user --upgrade \
    pip \
    setuptools \
    wheel \
    cython \
    ninja

########################################
# CLONE RESEARCH REPO
########################################
log "Preparing research workspace"

if [ ! -d "$WORKSPACE_DIR" ]; then
    git clone "$RESEARCH_REPO_URL" "$WORKSPACE_DIR"
else
    echo "Research workspace already exists."
    cd "$WORKSPACE_DIR"
    git pull
fi

cd "$WORKSPACE_DIR"

########################################
# CLONE TVM
########################################
log "Preparing TVM source"

if [ ! -d "$TVM_ROOT" ]; then
    git clone -b "$TVM_VERSION" --recursive \
        https://github.com/apache/tvm.git \
        "$TVM_ROOT"
else
    echo "TVM repo exists. Syncing..."
    cd "$TVM_ROOT"
    git fetch --tags


    git checkout "$TVM_VERSION"
    git submodule update --init --recursive
fi

########################################
# APPLY TVM MULTITHREADING PATCH
########################################
log "Applying multithreading patch"
cd "$TVM_ROOT"
git reset --hard HEAD  # Ensure a clean slate before patching
cat << 'EOF_PATCH' > tvm_multithreading.patch
diff --git a/src/runtime/thread_pool.cc b/src/runtime/thread_pool.cc
index 8d769fbe6..623eb71b6 100644
--- a/src/runtime/thread_pool.cc
+++ b/src/runtime/thread_pool.cc
@@ -261,6 +261,10 @@ class SpscTaskQueue {
 class ThreadPool {
  public:
   ThreadPool() : num_workers_(tvm::runtime::threading::MaxConcurrency()) {
+    const char* val = getenv("TVM_NUM_THREADS");
+    if (val != nullptr) {
+      num_workers_ = atoi(val);
+    }
     const char* exclude_worker0 = getenv("TVM_EXCLUDE_WORKER0");
     if (exclude_worker0 && atoi(exclude_worker0) == 0) {
       exclude_worker0_ = false;
@@ -342,7 +346,11 @@ class ThreadPool {
     threads_ = std::make_unique<tvm::runtime::threading::ThreadGroup>(
         num_workers_, [this](int worker_id) { this->RunWorker(worker_id); },
         exclude_worker0_ /* include_main_thread */);
-    num_workers_used_ = threads_->Configure(threading::ThreadGroup::kBig, 0, exclude_worker0_);
+    std::vector<unsigned int> cpus;
+    for (int i = 0; i < num_workers_; ++i) {
+        cpus.push_back(i);
+    }
+    num_workers_used_ = threads_->Configure(threading::ThreadGroup::kSpecifyThreadShareAllCore, num_workers_, exclude_worker0_, cpus);
   }
 
   // Internal worker function.
@@ -475,6 +483,10 @@ int32_t NumThreads() { return tvm::runtime::ThreadPool::ThreadLocal()->NumThread
 
 int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
   int num_workers = tvm::runtime::threading::MaxConcurrency();
+  const char* val = getenv("TVM_NUM_THREADS");
+  if (val != nullptr) {
+    num_workers = atoi(val);
+  }
   if (num_workers == 1) {
     std::atomic<int32_t> sync_counter{0};
     TVMParallelGroupEnv env;
diff --git a/src/runtime/threading_backend.cc b/src/runtime/threading_backend.cc
index ef835f20d..c4205524c 100644
--- a/src/runtime/threading_backend.cc
+++ b/src/runtime/threading_backend.cc
@@ -180,6 +180,9 @@ class ThreadGroup::Impl {
         num_workers_used = threading::MaxConcurrency();
     }
     // if a specific number was given, use that
+    if (nthreads == 0 && getenv("TVM_NUM_THREADS") != nullptr) {
+      nthreads = atoi(getenv("TVM_NUM_THREADS"));
+    }
     if (nthreads) {
       num_workers_used = nthreads;
     }
@@ -189,6 +192,7 @@ class ThreadGroup::Impl {
     // ones.
     num_workers_used = std::min(num_workers_, num_workers_used);
     SetAffinity(exclude_worker0, mode);
+    LOG(INFO) << "[Configure] ThreadPool Configured - Mode: " << mode << ", num_workers_used: " << num_workers_used;
     return num_workers_used;
   }
 
@@ -200,7 +204,9 @@ class ThreadGroup::Impl {
 #ifndef __hexagon__
     const char* val = getenv("TVM_BIND_THREADS");
     if (val != nullptr && atoi(val) != 1) {
-      return;
+      if (mode != kSpecifyThreadShareAllCore) {
+        return;
+      }
     }
     // Do not set affinity if there are more workers than found cores and mode is not kSpecify*.
     if (sorted_order_.size() < static_cast<unsigned int>(num_workers_)) {
@@ -394,20 +400,17 @@ void SetMaxConcurrency(int value) {
 }
 int MaxConcurrency() {
   int max_concurrency = 1;
-  if (tvm::runtime::threading::max_concurrency != 0) {
+  const char* val = getenv("TVM_NUM_THREADS");
+  if (val == nullptr) {
+    val = getenv("OMP_NUM_THREADS");
+  }
+  if (val != nullptr) {
+    max_concurrency = atoi(val);
+  } else if (tvm::runtime::threading::max_concurrency != 0) {
     max_concurrency = tvm::runtime::threading::max_concurrency;
   } else {
-    const char* val = getenv("TVM_NUM_THREADS");
-    if (val == nullptr) {
-      val = getenv("OMP_NUM_THREADS");
-    }
-    if (val != nullptr) {
-      max_concurrency = atoi(val);
-    } else {
-      max_concurrency = std::thread::hardware_concurrency();
-#if defined(_M_X64) || defined(__x86_64__)
-      max_concurrency /= 2;  // ignore hyper-threading
-#elif defined(__hexagon__)
+    max_concurrency = std::thread::hardware_concurrency();
+#if defined(__hexagon__)
       // Ideally max_concurrency is set to the total count of 128B
       // HVX units available. This prevenets threads unable to lock
       // an HVX unit from scheduling work on the Scalar cores instead
@@ -430,7 +433,6 @@ int MaxConcurrency() {
         max_concurrency = std::min(num_hvx128_contexts, max_concurrency);
       }
 #endif
-    }
   }
   return std::max(max_concurrency, 1);
 }
EOF_PATCH
patch -p1 < tvm_multithreading.patch

cd "$TVM_ROOT"

########################################
# CLEAN BUILD
########################################
log "Creating clean build directory"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

########################################
# CONFIGURE TVM
########################################
log "Generating config.cmake"

cp ../cmake/config.cmake .

cat >> config.cmake <<EOF

set(USE_LLVM ${LLVM_CONFIG})
set(USE_RPC ${ENABLE_RPC})
set(USE_GRAPH_EXECUTOR ${ENABLE_GRAPH_EXECUTOR})
set(USE_PROFILER ${ENABLE_PROFILER})
set(USE_OPENMP ${ENABLE_OPENMP})
EOF

########################################
# CMAKE CONFIGURE
########################################
log "Running CMake configure"

cmake .. -G "${GENERATOR}"

########################################
# BUILD TVM
########################################
log "Building TVM"

ninja -j"${JOBS}"

########################################
# CREATE VENV
########################################
log "Creating virtual environment"

cd "$WORKSPACE_DIR"

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

########################################
# PERSIST ENV VARIABLES IN VENV
########################################
log "Setting persistent environment variables in venv"

if ! grep -q "TVM_HOME=" "$VENV_DIR/bin/activate"; then
    echo "export TVM_HOME=\"$TVM_ROOT\"" >> "$VENV_DIR/bin/activate"
    echo "export PYTHONPATH=\"$TVM_ROOT/python:$WORKSPACE_DIR:\${PYTHONPATH:-}\"" >> "$VENV_DIR/bin/activate"
    echo "export LD_LIBRARY_PATH=\"$TVM_ROOT/build:$TVM_ROOT/build/lib:\${LD_LIBRARY_PATH:-}\"" >> "$VENV_DIR/bin/activate"
else
    # Ensure previously written lines are compatible with nounset mode.
    sed -i -E "s|^export TVM_HOME=.*$|export TVM_HOME=\"$TVM_ROOT\"|" "$VENV_DIR/bin/activate"
    sed -i -E "s|^export PYTHONPATH=.*$|export PYTHONPATH=\"$TVM_ROOT/python:$WORKSPACE_DIR:\${PYTHONPATH:-}\"|" "$VENV_DIR/bin/activate"
    sed -i -E "s|^export LD_LIBRARY_PATH=.*$|export LD_LIBRARY_PATH=\"$TVM_ROOT/build:$TVM_ROOT/build/lib:\${LD_LIBRARY_PATH:-}\"|" "$VENV_DIR/bin/activate"
fi

set +u
source "$VENV_DIR/bin/activate"
set -u

########################################
# INSTALL Python Packages (TVM Editable + Research dependencies)
########################################
log "Installing Packages"

# Install TVM python bindings as an editable install
# This binds it directly to the source folder without redundantly copying files.
cd "$TVM_ROOT/python"
$PIP_CMD install -e .

cd "$WORKSPACE_DIR"


########################################
# VALIDATE TVM
########################################
log "Validating TVM installation"

$PYTHON_BIN - <<EOF
import tvm
print("TVM OK")
print("Version:", tvm.__version__)
EOF

########################################
# INSTALL REQUIREMENTS
########################################
log "Installing requirements"

$PIP_CMD install --upgrade pip setuptools wheel

if [ -f "$WORKSPACE_DIR/requirements.txt" ]; then
    $PIP_CMD install -r "$WORKSPACE_DIR/requirements.txt"
else
    echo "No requirements.txt found. Skipping."
fi

########################################
# DONE
########################################
log "Setup complete"

echo
echo "Final layout:"
echo "$WORKSPACE_DIR"
echo "├── tvm"
echo "├── venv"
echo "├── research"
echo "├── scripts"
echo "└── requirements.txt"

########################################
# OPEN EXECUTION GUIDE (CLEAN)
########################################
log "Opening execution guide"

open_url() {
    local url="$1"

    # Prefer WSL browser bridge
    if command -v wslview >/dev/null 2>&1; then
        nohup wslview "$url" >/dev/null 2>&1 &
        return 0
    fi

    # Standard Linux opener
    if command -v xdg-open >/dev/null 2>&1; then
        nohup xdg-open "$url" >/dev/null 2>&1 &
        return 0
    fi

    # macOS fallback (just in case someone runs it there)
    if command -v open >/dev/null 2>&1; then
        nohup open "$url" >/dev/null 2>&1 &
        return 0
    fi

    return 1
}

if open_url "$EXECUTION_GUIDE_URL"; then
    echo "Execution guide opened in browser."
else
    echo "Could not auto-open browser."
    echo "Open manually:"
    echo "$EXECUTION_GUIDE_URL"
fi
