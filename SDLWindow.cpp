#include "SDLWindow.h"

#include <iostream>

#include <GL/glew.h>

#include "CudaMath.h"
#include "DisplayItem.h"
#include "FractalObject.h"
#include "RayMarchingTexture.h"
#include "Util.h"

using namespace std;

const float FOVy = 35.f;
const float zNear = 0.1f;
const float zFar = 300.f;

SDLWindow::SDLWindow()
    :
      bIsValidWindow_(false),
      bDoContinue_(true),
      Width_(1200),
      Height_(800),
      bRotateCamera_(false)
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        Util::LogSDLError(cerr, "SDL_init");
    }

    Window_ = SDL_CreateWindow( "Yayy!", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, Width_, Height_,
                                SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE );

    if (Window_ == NULL)
    {
        Util::LogSDLError(cerr, "SDL_CreateWindow");
    }

    Context_ = SDL_GL_CreateContext(Window_);

    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    glewInit();

    bIsValidWindow_ = true;

    InitShaders();
    InitGL();
    InitScene();

}

SDLWindow::~SDLWindow()
{
    bIsValidWindow_ = false;
    SDL_DestroyWindow(Window_);
    SDL_Quit();

    for (uint i = 0; i < SceneObjects_.size(); i++)
    {
        delete SceneObjects_[i];
    }
}

void SDLWindow::Animate()
{
    Camera_.Move();
//    Fractal_->Update();
    Fractal2_->SetCameraInfo(Camera_.GetPosition(), Camera_.GetDirection(), Camera_.GetUp());
    Fractal2_->Update();
    Camera_.AdjustMoveSpeedFactor(Fractal2_->GetDistanceFromCamera());
}

void SDLWindow::Draw()
{
    if (!bIsValidWindow_)
    {
        cerr << "Not a valid window!!!" << endl;
        return;
    }

    glClearColor(0.4f, 0.4f, 0.4f, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, Width_, Height_);

    Camera_.SetMatrix(&VisMatrix_);
    SetProjection();
//    Light_.SetPosition(Camera_.GetPosition() + Camera_.GetDirection());

    for (auto it = SceneObjects_.begin(); it != SceneObjects_.end(); it++)
    {
        (*it)->Draw();
    }

//    glColor4f(1.f, 1.f, 1.f, 1.f);
//    glPointSize(20.f);
//    glBegin(GL_POINTS);
//    glVertex3fv((float*)&Light_.GetPosition());
//    glEnd();

    // Draw HUD
    {
        const uint HUDWidth = Width_;
        const uint HUDHeight = Height_;
        const uint2 HUDPos = make_uint2(0, Height_ - HUDHeight);
        glViewport(HUDPos.x, HUDPos.y, HUDWidth, HUDHeight);
        HUD_->Draw();
    }

    SDL_GL_SwapWindow(Window_);
}

void SDLWindow::HandleEvents()
{
    SDL_Event Event;
    while (SDL_PollEvent(&Event))
    {
        switch (Event.type) {
        case SDL_QUIT:
            bDoContinue_ = false;
            break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
            HandleKeyPress(Event.key.keysym.sym, (Event.type == SDL_KEYDOWN));
            break;
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            HandleMouseClick(Event.button.button, Event.button.state, Event.button.x, Event.button.y);
            break;
        case SDL_MOUSEMOTION:
            HandleMouseMove(Event.motion.x, Event.motion.y);
            break;
        case SDL_WINDOWEVENT:
            if (Event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
            {
                ResizeWindow(Event.window.data1, Event.window.data2);
            }
            else
            {
                int w, h;
                SDL_GetWindowSize( Window_, &w, &h);
                ResizeWindow(w, h);
            }
        default:
            break;
        }
    }
}

void SDLWindow::HandleKeyPress(SDL_Keycode Key, bool bPress)
{
    switch (Key) {
    case SDLK_ESCAPE:
    case SDLK_q:
        bDoContinue_ = false;
        break;
    case SDLK_w:
    case SDLK_UP:
        Camera_.SetMoveForward(bPress);
        break;
    case SDLK_a:
    case SDLK_LEFT:
        Camera_.SetMoveLeft(bPress);
        break;
    case SDLK_s:
    case SDLK_DOWN:
        Camera_.SetMoveBack(bPress);
        break;
    case SDLK_PAGEUP:
        Camera_.SetMoveUp(bPress);
        break;
    case SDLK_PAGEDOWN:
        Camera_.SetMoveDown(bPress);
        break;
    case SDLK_d:
    case SDLK_RIGHT:
        Camera_.SetMoveRight(bPress);
        break;
    case SDLK_z:
        if (bPress) Fractal_->MovePlane(1);
        break;
    case SDLK_x:
        if (bPress) Fractal_->MovePlane(-1);
        break;
    default:
        //cout << "Pressed " << Key << endl;
        break;
    }
}

void SDLWindow::HandleMouseClick(int Button, int State, int x, int y)
{
    bool bPressed = (State == SDL_PRESSED);

    switch (Button)
    {
    case SDL_BUTTON_LEFT:
        if (bPressed)
        {
            PreviousX_ = x;
            PreviousY_ = y;
            bRotateCamera_ = true;
        }
        else
        {
            bRotateCamera_ = false;
        }
        break;
    case SDL_BUTTON_RIGHT:
        break;
    default:
        cerr << "Unknown mouse button pressed" << endl;
        break;
    }
}

void SDLWindow::HandleMouseMove(int x, int y)
{
    if (bRotateCamera_)
    {
        int Horizontal = x - PreviousX_;
        int Vertical = y - PreviousY_;
        Camera_.Rotate(Horizontal, Vertical);
        PreviousX_ = x;
        PreviousY_ = y;
    }
}

void SDLWindow::ResizeWindow(int NewX, int NewY)
{
    Width_ = NewX;
    Height_ = NewY;
    glViewport(0, 0, (int)Width_, (int)Height_);
}

void SDLWindow::SetProjection()
{
    ProjMatrix_.Perspective(FOVy, Width_ / Height_, zNear, zFar);
    Fractal2_->SetPerspective(FOVy, Width_ / Height_, zNear, zFar);
}

void SDLWindow::InitShaders()
{
    BaseShaders_.ReadShaderSource("BaseVertexShader.glsl", GL_VERTEX_SHADER);
    BaseShaders_.ReadShaderSource("BaseFragmentShader.glsl", GL_FRAGMENT_SHADER);
    BaseShaders_.LinkProgram();

    PhongShader_.ReadShaderSource("PhongLightingVertexShader.glsl", GL_VERTEX_SHADER);
    PhongShader_.ReadShaderSource("PhongLightingFragmentShader.glsl", GL_FRAGMENT_SHADER);
    PhongShader_.LinkProgram();
}

void SDLWindow::InitGL()
{
    glClearColor(0.4f, 0.4f, 0.4f, 1.0);
    glEnable( GL_DEPTH_TEST );
}

void SDLWindow::InitScene()
{

    /*{
        DisplayItem* TestObject = new DisplayItem();

        float Vertices[] = {
            -1.f, 0.f, 0.f,
            0.f, 0.f, 0.f,
            -1.f, 1.f, 0.f,
            -0.5f, 0.5f, 1.f
        };

        uint Connect[] = {
            0, 1, 2,
            0, 1, 3,
            1, 2, 3,
            2, 0, 3
        };

        float Normals[] = {
            -1.f, 0.f, 0.1f,
            1.f, 0.f, 0.1f,
            0.f, 1.f, 0.1f,
            0.f, 0.f, 1.f
        };

        float4 Color = make_float4(0.f, 1.f, 0.f, 1.f);

        TestObject->Init(&PhongShader_, &ProjMatrix_, &VisMatrix_,
                         (float3*)Vertices, (float3*)Normals, 4, (uint3*)Connect, 4, &Color, 1,
                         NULL, NULL);
        SceneObjects_.push_back(TestObject);

        DisplayItem* T2 = new DisplayItem();

        float v2[] = {
            1.f, 0.f, 0.f,
            2.f, 0.f, 0.f,
            1.f, 1.f, 0.f
        };

        float4 c2 = make_float4(0.f, 0.f, 1.f, 1.f);
        T2->Init(&BaseShaders_, &ProjMatrix_, &VisMatrix_,
                 (float3*)v2, (float3*)Normals, 3, (uint3*)Connect, 1, &c2, 1,
                 NULL, NULL);
        SceneObjects_.push_back(T2);
    }*/


    {
        HUD_ = new DisplayItem();
        float Vertices[] = {
            -1.f, -1.f, 1.f,
             1.f, -1.f, 1.f,
             1.f,  1.f, 1.f,
            -1.f,  1.f, 1.f
        };
        uint Connect[] = {
            0, 1, 2,
            2, 3, 0
        };
        float4 Color = make_float4(1.f, 1.f, 1.f, 1.f);

        TransformMatrix* HUDProj = new TransformMatrix();
        HUDProj->Ortho(-1.f, 1.f, -1.f, 1.f, 1.f, -1.f);
        TransformMatrix* HUDVis = new TransformMatrix();
        HUDVis->LookAt(0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f);

        SDL_Surface* TexImage = SDL_LoadBMP("color16.bmp");

        float TexCoord[] = {
            0.f, 1.f,
            1.f, 1.f,
            1.f, 0.f,
            0.f, 0.f
        };

        HUD_->Init(&BaseShaders_, HUDProj, HUDVis, (float3*)Vertices, NULL, 4, (uint3*)Connect, 2, &Color, 1, TexImage, (float2*)TexCoord);

        {
            Fractal_ = new FractalObject();
            Fractal_->Init(&PhongShader_, &ProjMatrix_, &VisMatrix_,
                           NULL, NULL, 100000, NULL, 1500000, NULL, 100000,
                           NULL, NULL);
            //Fractal_->AttachGLTexture(HUD_->GetTextureId(), TexImage->w * TexImage->h * 4 * sizeof(GLubyte));
            HUD_->SetTexture(Fractal_->GetTextureId());
//            SceneObjects_.push_back(Fractal_);
        }

        {
            Fractal2_ = new RayMarchingTexture();
            Fractal2_->Init();
            HUD_->SetTexture(Fractal2_->GetTextureId());
        }

    }

    PhongShader_.SetLightSource(&Light_);
    PhongShader_.SetMaterial(&BaseMaterial_);
}
