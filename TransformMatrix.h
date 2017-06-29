#ifndef TRANSFORMMATRIX_H
#define TRANSFORMMATRIX_H

#include <iostream>
#include <stack>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

/**
 * @brief A simple stack of matrices on which we can apply various transforms used in OpenGL.
 *
 *  This class is taken from a code sample from the INF2705 course at École polytechnique de Montréal.
 */

class TransformMatrix
{
public:
    TransformMatrix(std::string Name = "SomeName")
        : Name_(Name)
    {
        Matrices_.push(glm::mat4(1.f));
    }

    void LoadIdentity() { Matrices_.top() = glm::mat4(1.f); }

    void Scale(GLfloat X, GLfloat Y, GLfloat Z) { Matrices_.top() = glm::scale(Matrices_.top(), glm::vec3(X, Y, Z)); }
    void Translate(GLfloat X, GLfloat Y, GLfloat Z)
    {
        Matrices_.top() = glm::translate(Matrices_.top(), glm::vec3(X, Y, Z));
    }
    void Rotate(GLfloat Angle, GLfloat X, GLfloat Y, GLfloat Z)
    {
        Matrices_.top() = glm::rotate(Matrices_.top(), (GLfloat)glm::radians(Angle), glm::vec3(X, Y, Z));
    }

    void LookAt(GLfloat ObsX, GLfloat ObsY, GLfloat ObsZ, GLfloat TargetX, GLfloat TargetY, GLfloat TargetZ,
                GLfloat UpX, GLfloat UpY, GLfloat UpZ)
    {
        Matrices_.top() =
          glm::lookAt(glm::vec3(ObsX, ObsY, ObsZ), glm::vec3(TargetX, TargetY, TargetZ), glm::vec3(UpX, UpY, UpZ));
    }
    void Frustum(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far)
    {
        Matrices_.top() = glm::frustum(Left, Right, Bottom, Top, Near, Far);
    }
    void Perspective(GLfloat FovY, GLfloat Aspect, GLfloat Near, GLfloat Far)
    {
        Matrices_.top() = glm::perspective(glm::radians(FovY), Aspect, Near, Far);
    }
    void Ortho(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top, GLfloat Near, GLfloat Far)
    {
        Matrices_.top() = glm::ortho(Left, Right, Bottom, Top, Near, Far);
    }
    void Ortho2D(GLfloat Left, GLfloat Right, GLfloat Bottom, GLfloat Top)
    {
        Matrices_.top() = glm::ortho(Left, Right, Bottom, Top);
    }

    void PushMatrix() { Matrices_.push(Matrices_.top()); }
    void PopMatrix() { Matrices_.pop(); }

    const glm::mat4 GetMatrix() const { return Matrices_.top(); }
    const glm::mat4 SetMatrix(glm::mat4 NewMatrix) { return (Matrices_.top() = NewMatrix); }

    operator const GLfloat*() const { return glm::value_ptr(Matrices_.top()); }

    friend std::ostream& operator<<(std::ostream& o, const TransformMatrix& Tm)
    {
        // return o << glm::to_string(mp.matr_.top());
        glm::mat4 m = Tm.Matrices_.top(); // o.precision(3); o.width(6);
        return o << std::endl
                 << "   " << m[0][0] << " " << m[1][0] << " " << m[2][0] << " " << m[3][0] << std::endl
                 << "   " << m[0][1] << " " << m[1][1] << " " << m[2][1] << " " << m[3][1] << std::endl
                 << "   " << m[0][2] << " " << m[1][2] << " " << m[2][2] << " " << m[3][2] << std::endl
                 << "   " << m[0][3] << " " << m[1][3] << " " << m[2][3] << " " << m[3][3] << std::endl;
    }

private:
    std::stack<glm::mat4> Matrices_;
    std::string           Name_;
};

#endif // TRANSFORMMATRIX_H
