#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>

#include <SDL2/SDL.h>

void LogSDLError(std::ostream &OutStream, const std::string &Message)
{
    OutStream << Message << " error: " << SDL_GetError() << std::endl;
}

#endif // UTIL_H
