TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += \
    /opt/cuda/include

SOURCES += \
    Main.cpp \
    SDLWindow.cpp \
    TransformMatrix.cpp \
    ShaderProgram.cpp \
    DisplayItem.cpp \
    Camera.cpp \
    LightSource.cpp \
    LightShader.cpp \
    Material.cpp

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
    Material.h

unix: LIBS += -lSDL2 \
              -lGLEW \
              -lGL

DISTFILES += \
    BaseVertexShader.glsl \
    BaseFragmentShader.glsl \
    PhongLightingVertexShader.glsl \
    PhongLightingFragmentShader.glsl
