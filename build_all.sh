DIR=$(pwd)

# 0. Install Boost and Eigen3 for mycpp (required on first run in container; no-op if already installed)
echo "Ensuring Boost and Eigen3 are installed for mycpp..."
apt-get update -qq && apt-get install -y -qq \
  libboost-dev libboost-system-dev libboost-program-options-dev \
  libeigen3-dev

# 1. Build mycpp
cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11

# 2. Kaolin: install without deps to avoid usd-core (no PyPI wheel on many platforms)
cd /kaolin && rm -rf build *egg* && pip install --no-deps -e .

cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
