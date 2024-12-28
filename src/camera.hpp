#pragma once

#include "geom.hpp"

#include <SFML/Window.hpp>

struct Camera {
  vec3 position;
  rotation orientation;

  // Perspective parameters
  float fov;         // Field of view in radians
  float aspectRatio; // width/height
  float nearPlane;   // Near clipping plane
  float farPlane;    // Far clipping plane

  Camera(float fovDegrees = 60.0f, float aspect = 16.0f / 9.0f,
         float near = 0.1f, float far = 100.0f)
      : position(0, 0, 0), orientation() // Identity rotation
        ,
        fov(fovDegrees * M_PI / 180.0f), aspectRatio(aspect), nearPlane(near),
        farPlane(far) {}

  // Transform a point from world space to screen space [-1,1] range
  vec3 worldToScreen(const vec3 &worldPos) const {
    // 1. Transform to camera space (view transform)
    vec3 viewPos = orientation.inverse() * (worldPos - position);

    // 2. Perspective projection
    float tanHalfFov = tanf(fov * 0.5f);
    float y = -(viewPos.y / (viewPos.z * tanHalfFov));
    float x = -(viewPos.x / (viewPos.z * tanHalfFov * aspectRatio));

    // Z is transformed to [0,1] range for depth buffer
    float z = (viewPos.z - nearPlane) / (farPlane - nearPlane);

    return {x, y, z};
  }

  // Transform screen space point [-1,1] to window coordinates
  vec3 screenToWindow(const vec3 &screenPos, float windowWidth,
                      float windowHeight) const {
    return {
        (screenPos.x + 1.0f) * 0.5f * windowWidth,
        (screenPos.y + 1.0f) * 0.5f * windowHeight,
        screenPos.z // Preserve depth
    };
  }

  // Combined transform from world to window coordinates
  vec3 worldToWindow(const vec3 &worldPos, float windowWidth,
                     float windowHeight) const {
    vec3 screenPos = worldToScreen(worldPos);
    return screenToWindow(screenPos, windowWidth, windowHeight);
  }

  void lookAt(const vec3 &target) {
    vec3 forward = (target - position).normalized();
    vec3 up = vec3(0, 1, 0);

    // Handle case where we're looking straight up/down
    if (fabsf(forward.dot(up)) > 0.999f) {
      up = vec3(0, 0, forward.y > 0 ? -1 : 1);
    }

    vec3 right = up.cross(forward).normalized();
    up = forward.cross(right);

    // Convert orthonormal basis to quaternion
    float trace = right.x + up.y + forward.z;

    if (trace > 0) {
      float s = sqrtf(trace + 1.0f) * 2.0f;
      orientation = {(up.z - forward.y) / s, (forward.x - right.z) / s,
                     (right.y - up.x) / s, s * 0.25f};
    } else if (right.x > up.y && right.x > forward.z) {
      float s = sqrtf(1.0f + right.x - up.y - forward.z) * 2.0f;
      orientation = {s * 0.25f, (right.y + up.x) / s, (forward.x + right.z) / s,
                     (up.z - forward.y) / s};
    } else if (up.y > forward.z) {
      float s = sqrtf(1.0f + up.y - right.x - forward.z) * 2.0f;
      orientation = {(right.y + up.x) / s, s * 0.25f, (up.z + forward.y) / s,
                     (forward.x - right.z) / s};
    } else {
      float s = sqrtf(1.0f + forward.z - right.x - up.y) * 2.0f;
      orientation = {(forward.x + right.z) / s, (up.z + forward.y) / s,
                     s * 0.25f, (right.y - up.x) / s};
    }
  }
};


struct FirstPersonMouse {
  sf::Window *window;
  float sensitivity = 0.02f;
  float yaw = 0;
  float pitch = 0;

  FirstPersonMouse(sf::Window *window) : window(window) {

    window->setMouseCursorVisible(false);
  }

  void update() {
    sf::Vector2i center(window->getPosition().x + window->getSize().x / 2,
                        window->getPosition().y + window->getSize().y / 2);

    sf::Vector2i mouseDelta = sf::Mouse::getPosition() - center;

    float mouseDeltaX = mouseDelta.x;
    float mouseDeltaY = mouseDelta.y;

    float len = sqrtf(mouseDeltaX * mouseDeltaX + mouseDeltaY * mouseDeltaY);


    if (len > 100.0f) {
      float factor = 100.0f / len;
      mouseDeltaX *= factor;
      mouseDeltaY *= factor;
    }

    yaw += mouseDeltaX  * sensitivity;
    pitch += mouseDeltaY * sensitivity;

    sf::Mouse::setPosition(center);
  }

  rotation getOrientation() {
    return rotation::fromAxisAngle({0, 1, 0}, yaw * M_PI / 180.0f) *
           rotation::fromAxisAngle({1, 0, 0}, pitch * M_PI / 180.0f);
  }
};

struct FirstPersonController {
  Camera cam;
  FirstPersonMouse mouse;

  FirstPersonController(sf::Window *window)
      : cam(60.0f, 16.0f / 9.0f, 0.1f, 100.0f), mouse(window) {
    cam.position = {0, 0, 0};
  }

  void update(float dt) {

    const float speed = 5.0f * cam.position.length();

    vec3 velocity = {0, 0, 0};


    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
      velocity.z += 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
      velocity.z -= 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
      velocity.x -= 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
      velocity.x += 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
      velocity.y += 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
      velocity.y -= 1;
    }

    cam.position += cam.orientation * velocity.normalized() * speed * dt;

    mouse.update();

    cam.orientation = mouse.getOrientation();

    // rb.orientation = cam.orientation;
  }
};

