TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    Main.cpp \
    SDLWindow.cpp \
    TransformMatrix.cpp \
    ShaderProgram.cpp \
    DisplayItem.cpp \
    Camera.cpp \
    LightSource.cpp \
    LightShader.cpp \
    Material.cpp \
    FractalObject.cpp \
    Util.cpp \
    TexturedItem.cpp \
    RayMarchingTexture.cpp

HEADERS += \
    Util.h \
    SDLWindow.h \
    TransformMatrix.h \
    ShaderProgram.h \
    DisplayItem.h \
    Common.h \
    Camera.h \
    CudaMath.h \
    LightSource.h \
    LightShader.h \
    Material.h \
    FractalObject.h \
    HostDeviceCode.h \
    Kernels.h \
    DeviceUtil.cuh \
    DistanceEstimators.cuh \
    MarchingCubesTables.h \
    TexturedItem.h \
    RayMarchingTexture.h \
    BasicTimer.h

unix: LIBS += -lSDL2 \
              -lGLEW \
              -lGL

DISTFILES += \
    BaseVertexShader.glsl \
    BaseFragmentShader.glsl \
    PhongLightingVertexShader.glsl \
    PhongLightingFragmentShader.glsl \
    Bucky.raw \
    black.bmp \
    Kernels.cu


DEPENDPATH += /opt/cuda/lib64

##############
# CUDA stuff #
##############
CUDA_SOURCES += Kernels.cu

CUDA_DIR = "/opt/cuda"
CUDA_OBJECTS_DIR = ./
CUDA_ARCH = compute_30
CUDA_CODE = sm_61

SYSTEM_TYPE = 64

INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_DIR/samples/common/inc

QMAKE_LIBDIR += $$CUDA_DIR/lib64/
LIBS += -lcudart

NVCC_OPTIONS += -g

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = \
        $$CUDA_DIR/bin/nvcc \
        -D_DEBUG \
        $$NVCC_OPTIONS \
        $$CUDA_INC \
        $$NVCC_LIBS \
        --machine $$SYSTEM_TYPE --gpu-architecture=$$CUDA_ARCH --gpu-code=$$CUDA_CODE \
        -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE --gpu-architecture=$$CUDA_ARCH --gpu-code=$$CUDA_CODE -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
