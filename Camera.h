#ifndef CAMERA_H
#define CAMERA_H

#include "Common.h"
#include "TransformMatrix.h"

class Camera
{
public:
    Camera();

    void SetMatrix(TransformMatrix* Matrix);

    float3 GetPosition() const { return Position_; }
    float3 GetDirection() const { return Direction_; }
    float3 GetUp() const { return Up_; }

    void Move();
    void SetMoveForward(bool bForward = true) { bMoveForward_ = bForward; }
    void SetMoveBack(bool bBack = true) { bMoveBack_ = bBack; }
    void SetMoveLeft(bool bLeft = true) { bMoveLeft_ = bLeft; }
    void SetMoveRight(bool bRight = true) { bMoveRight_ = bRight; }
    void SetMoveUp(bool bUp = true) { bMoveUp_ = bUp; }
    void SetMoveDown(bool bDown = true) { bMoveDown_ = bDown; }
    void SetTurnLeft(bool bLeft = true) { bTurnLeft_ = bLeft; }
    void SetTurnRight(bool bRight = true) { bTurnRight_ = bRight; }

    void Rotate(int Horizontal, int Vertical);

    void AdjustMoveSpeedFactor(const float Distance);

private:
    float3 Position_;
    float3 Direction_;
    float3 Up_;

    bool bMoveForward_;
    bool bMoveBack_;
    bool bMoveLeft_;
    bool bMoveRight_;
    bool bMoveUp_;
    bool bMoveDown_;
    bool bTurnLeft_;
    bool bTurnRight_;

    float MoveSpeed_;
    float MoveSpeedFactor_;
    float RotateSpeed_;

    float OldDistance_;
};

#endif // CAMERA_H
