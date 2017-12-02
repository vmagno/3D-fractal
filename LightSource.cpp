#include "LightSource.h"

#include <GL/glew.h>

void LightSource::Draw()
{
    glPointSize(10);
    glBegin(GL_POINTS);
    glVertex3fv((const GLfloat*)&Position_);
    glEnd();
}
