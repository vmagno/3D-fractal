#include <iostream>

#include "SDLWindow.h"

using namespace std;

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        for (int i = 0; i < argc; i++)
        {
            cout << i << ": " << argv[i] << endl;
        }
    }

    SDLWindow Window;

    while (Window.DoContinue())
    {
        Window.HandleEvents();
        Window.Animate();
        Window.Draw();
    }

    cout << "It's done." << endl;

    return 0;
}
