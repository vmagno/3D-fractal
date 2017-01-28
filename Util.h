#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <iostream>
#include <string>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Common.h"

class Util
{
public:
    static void LogSDLError(std::ostream &OutStream, const std::string &Message);
    static void DumpDeviceArray(uint* DeviceArray, uint3 Dimension, std::string Filename);

    /**
     * @brief LoadBMP. Taken from http://www.opengl-tutorial.org/beginners-tutorials/tutorial-5-a-textured-cube/
     * @param Filename
     */
    static GLuint LoadBMP(const char* Filename);

};

#endif // UTIL_H
