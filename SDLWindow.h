#ifndef SDLWINDOW_H
#define SDLWINDOW_H

#include <vector>

#include <SDL2/SDL.h>

#include "Camera.h"
#include "DisplayItem.h"
#include "LightShader.h"
#include "LightSource.h"
#include "Material.h"
#include "ShaderProgram.h"
#include "TransformMatrix.h"

class SDLWindow
{
public:
    SDLWindow();
    ~SDLWindow();

    bool DoContinue() { return bDoContinue_; }

    void Animate();
    void Draw();
    void HandleEvents();



    void SetProjection();

private:
    bool bIsValidWindow_;
    bool bDoContinue_;
    SDL_Window* Window_;
    SDL_GLContext Context_;
    float Width_;
    float Height_;
    TransformMatrix ProjMatrix_;
    TransformMatrix VisMatrix_;
    Camera Camera_;
    LightSource Light_;
    Material BaseMaterial_;

    ShaderProgram BaseShaders_;
    LightShader PhongShader_;

    bool bRotateCamera_;
    int PreviousX_, PreviousY_;

    std::vector<DisplayItem*> SceneObjects;

    /**
     * @brief HandleKeyPress
     * @param Key
     * @param bPress Whether the key is being pressed or released (true for pressed)
     */
    void HandleKeyPress(SDL_Keycode Key, bool bPress);

    void HandleMouseClick(int Button, int State, int x, int y);
    void HandleMouseMove(int x, int y);

    void ResizeWindow(int NewX, int NewY);

    void InitShaders();
    void InitGL();
    void InitScene();

};

#endif // SDLWINDOW_H
