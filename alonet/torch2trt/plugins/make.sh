# to be run from root of aloception
# add commands for each plugin here for automatic build

ALONET_ROOT=$1

# ====== ms_deform_im2col
BUILD_DIR=$ALONET_ROOT/torch2trt/plugins/ms_deform_im2col/build
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake .. -DTRT_LIB=${TRT_LIBPATH}/lib/ -DTRT_INCLUDE=${TRT_LIBPATH}/include/
make -j
cd -
