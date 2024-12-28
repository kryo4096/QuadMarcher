#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/Window.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "buffer.hpp"
#include "geom.hpp"
#include "raymarching.hpp"
#include "camera.hpp"



class SphereScene : public RayMarchableScene {
  float f(const vec3 &p) override { return p.length() - 1; }
  float maxRadius() override { return 1.0f; }
};

class MandelbulbScene : public RayMarchableScene {
  float f(const vec3 &p) override {
    vec3 z = p;
    float dr = 1.0;
    float r = 0.0;

    rotation rot = rotation::fromAxisAngle({0, 0, 1}, omp_get_wtime() / 10.0f);

    for (int i = 0; i < 6; i++) {
      r = z.length();

      if (r > 2.0)
        break;


      // Convert to polar coordinates
      float theta = acosf(z.z / r);
      float phi = atan2f(z.y, z.x);
      dr = powf(r, 7.0f) * 8.0f * dr + 1.0;

      // Scale and rotate the point
      float zr = powf(r, 8.0f);
      theta = theta * 8.0f;
      phi = phi * 8.0f;

      // Convert back to cartesian coordinates
      z = vec3(sinf(theta) * cosf(phi), sinf(phi) * sinf(theta), cosf(theta)) *
          zr;
      z += p;

      z = rot * z;
    }

    return 0.5f * logf(r) * r / dr;
  }

  float maxRadius() override { return 2.0f; }
};

int main() {
  sf::VideoMode videoMode = sf::VideoMode(1920, 1200);

  sf::RenderWindow window(videoMode, "Raymarching Test", sf::Style::Fullscreen);
  window.setFramerateLimit(60);

  CPUFramebuffer framebuffer(window);

  FirstPersonController player(&window);
  player.cam.position = {0, 0, -5};

  float t = 0;

  MandelbulbScene scene;
  RayMarcher raymarcher(&framebuffer, &player.cam, &scene);

  // rb.addTorque({0.0, 1.0, 0.0});
  // rb.addTorque({0.0, 0.0, 1.0});
  while (window.isOpen()) {

    window.clear(sf::Color::Black);
    sf::Event event;

    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        window.close();
      }

      if (event.type == sf::Event::KeyPressed) {
        if (event.key.code == sf::Keyboard::Escape) {
          window.close();
        }
        if (event.key.code == sf::Keyboard::R) {
          raymarcher.toggleWireframe();
        }
      }
    }

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    // framebuffer.clear(sf::Color::Black);
    raymarcher.render();

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::cout << "Time difference = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;

    framebuffer.drawTo(window);

    window.display();

    t += 0.01f;
    player.update(0.01f);
  }

  return 0;
}