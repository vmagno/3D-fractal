#include "Camera.h"

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "CudaMath.h"

using namespace std;

Camera::Camera()
    :
      bMoveForward_(false),
      bMoveBack_(false),
      bMoveLeft_(false),
      bMoveRight_(false),
      bMoveUp_(false),
      bMoveDown_(false),
      bTurnLeft_(false),
      bTurnRight_(false),
      MoveSpeed_(0.15f),
      RotateSpeed_(0.002f)
{
    Position_ = make_float3(0.f, 0.f, 10.f);
    Direction_ = make_float3(0.f, 0.f, -1.f);
    Up_ = make_float3(0.f, 1.f, 0.f);

    Direction_ = Normalize(Direction_);
    Up_ = Normalize(Up_);
}

void Camera::SetMatrix(TransformMatrix* Matrix)
{
    Matrix->LookAt( Position_.x, Position_.y, Position_.z,
                    Direction_.x + Position_.x, Direction_.y + Position_.y, Direction_.z + Position_.z,
                    Up_.x, Up_.y, Up_.z );
}

void Camera::Move()
{
    if (bMoveForward_)
    {
        Position_ += Direction_ * MoveSpeed_;
    }
    else if (bMoveBack_)
    {
        Position_ -= Direction_ * MoveSpeed_;
    }

    if (bMoveLeft_)
    {
        Position_ += Cross(Up_, Direction_) * MoveSpeed_;
    }
    else if (bMoveRight_)
    {
        Position_ -= Cross(Up_, Direction_) * MoveSpeed_;
    }

    if (bMoveUp_)
    {
        Position_ += Up_ * MoveSpeed_;
    }
    else if (bMoveDown_)
    {
        Position_ -= Up_ * MoveSpeed_;
    }
}

void Camera::Rotate(int Horizontal, int Vertical)
{
    glm::vec3 Up(Up_.x, Up_.y, Up_.z);
    glm::vec3 Dir(Direction_.x, Direction_.y, Direction_.z);
    glm::vec3 Left = glm::cross(Up, Dir);

    Dir = glm::rotate(Dir, -Horizontal * RotateSpeed_, Up);
    Dir = glm::rotate(Dir, Vertical * RotateSpeed_, Left);
    Dir = glm::normalize(Dir);
    Direction_ = make_float3(Dir.x, Dir.y, Dir.z);
}
