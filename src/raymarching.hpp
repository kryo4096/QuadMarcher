#pragma once

#include "camera.hpp"
#include "geom.hpp"
#include "buffer.hpp"
#include "color.hpp"

#include <random>

class RayMarchableScene {
public:
  virtual float f(const vec3 &p) = 0;
  virtual vec3 normal(const vec3 &p) {
    const float epsilon = 0.01f;
    return vec3(f(p + vec3(epsilon, 0, 0)) - f(p - vec3(epsilon, 0, 0)),
                f(p + vec3(0, epsilon, 0)) - f(p - vec3(0, epsilon, 0)),
                f(p + vec3(0, 0, epsilon)) - f(p - vec3(0, 0, epsilon)))
        .normalized();
  }
  virtual float maxRadius() { return 1.0f; }
};

struct RayMarchingConfig {
  int maxIterations = 100;
  float epsilon = 1e-3;
  float minDistance = 1e-2;
  float maxDistance = 100.0;
  float refinementMultiplier = 1.5;
  float jitterAAMagnitude = 0.001;
  int maxRefinementLevel = 20;
  int minTileSize = 1;
  int numPartitionsX = 16;
  int numPartitionsY = 10;
};



class RayMarcher {
public:
  CPUFramebuffer *buffer;
  PixelBuffer<int> levelBufRead, levelBufWrite;

  RayMarchableScene *scene;
  Camera *cam;

  RayMarchingConfig config;

  bool wireframe = false;

  std::mt19937 rng;

  RayMarcher(CPUFramebuffer *buffer, Camera *cam, RayMarchableScene *scene, RayMarchingConfig config = {})
      : buffer(buffer), cam(cam), scene(scene),
        levelBufRead(buffer->getWidth(), buffer->getHeight()),
        levelBufWrite(buffer->getWidth(), buffer->getHeight()), config(config) {
    rng.seed(std::random_device()());

    levelBufRead.clear(0);
    levelBufWrite.clear(0);
  }

  struct RayMarchResult {
    float depth;
    vec3 normal;
    vec3 position;
    int numSteps;
    bool hit;
  };

  RayMarchResult raymarch(const vec3 &origin, const vec3 &direction, float maxDistance,
                 float epsilon, float minDistance) {
    float t = minDistance;

    vec3 d = direction.normalized();
    vec3 p = origin + d * minDistance;

    for (int i = 0; i < config.maxIterations; i++) {


      float dist = scene->f(p);

      if (dist < epsilon) {

        return RayMarchResult {
          .depth = t, .normal = scene->normal(p),
          .position = p, .numSteps = i, .hit = true
        };
      }

      vec3 dp = d * dist;
      p += dp;
      t += dp.length();
    }

    if (p.length() < scene->maxRadius()) {

      return RayMarchResult{.depth = t,
                            .normal = scene->normal(p),
                            .position = p,
                            .numSteps = 100,
                            .hit = true};
    } else
    {
      return RayMarchResult{.depth = INFINITY,
                            .normal = vec3(0, 0, 0),
                            .position = p,
                            .numSteps = 100,
                            .hit = false};
    }
  }

  int renderQuadrantRecursive(int x0, int y0, int x1, int y1, float prevDepth,
                              int level) {

    std::normal_distribution<float> jitterAAOffset(0.0f, config.jitterAAMagnitude);

    int w = x1 - x0;
    int h = y1 - y0;

    int xMid = (x0 + x1) / 2;
    int yMid = (y0 + y1) / 2;


    const float focalLength = 0.1f;
    float aperture = focalLength * sinf(cam->fov / 2.0f);

    vec3 screenPos = {(xMid - 0.5f * buffer->getWidth()) / buffer->getHeight(),
                      -(yMid - 0.5f * buffer->getHeight()) / buffer->getHeight(),
                      0};

    screenPos += vec3(jitterAAOffset(rng), jitterAAOffset(rng), 0);

    screenPos *= aperture;
    screenPos.z = focalLength;

    vec3 worldPos = (cam->orientation * screenPos) + cam->position;
    vec3 direction = (worldPos - cam->position).normalized();



    auto result = raymarch(cam->position, direction, cam->farPlane, 1e-3, 1e-2);

    sf::Color color = colormaps::viridis(1.0f - result.numSteps / (float)config.maxIterations);

    color = mix(color, sf::Color::White, 1.0f - expf(-0.1 * result.depth));

    sf::Color bg = colormaps::viridis((direction.x * 0.5f + 0.5f));

    color = mix(color, bg, 1.0f - expf(-0.01 * result.depth));



    //color = mix(color, sf::Color::Black, omp_get_thread_num() / 32.0f);

    //color = mix(color, sf::Color::Black, 0.5f * result.normal.x + 0.5f);

    float errorHeuristic =
        result.normal.cross(direction).lengthSq() * powf(2.0f, -2 * level);

    int maxNeighLevel = 0;

    maxNeighLevel =
        std::max(maxNeighLevel, levelBufRead.getPixelOrDefault(xMid - w, yMid));
    maxNeighLevel =
        std::max(maxNeighLevel, levelBufRead.getPixelOrDefault(xMid + w, yMid));
    maxNeighLevel =
        std::max(maxNeighLevel, levelBufRead.getPixelOrDefault(xMid, yMid - h));
    maxNeighLevel =
        std::max(maxNeighLevel, levelBufRead.getPixelOrDefault(xMid, yMid + h));

    int maxLevel = level;

    bool split = false;

    if (config.refinementMultiplier*result.numSteps - (1 << level) > 0 && result.hit) {
      split = true;
    }

    if(w < config.minTileSize || h < config.minTileSize) {
      split = false;
    }

    if ((split || maxNeighLevel - level > 1) && level < config.maxRefinementLevel) {

      maxLevel = std::max(maxLevel, renderQuadrantRecursive(x0, y0, xMid, yMid,
                                                            result.depth, level + 1));
      maxLevel = std::max(maxLevel, renderQuadrantRecursive(xMid, y0, x1, yMid,
                                                            result.depth, level + 1));
      maxLevel = std::max(maxLevel, renderQuadrantRecursive(x0, yMid, xMid, y1,
                                                            result.depth, level + 1));
      maxLevel = std::max(maxLevel, renderQuadrantRecursive(xMid, yMid, x1, y1,
                                                            result.depth, level + 1));

    } else {
      buffer->mixRect(x0, y0, x1, y1, color, 0.5f);
      
      if (wireframe || !result.hit) {
        buffer->fillRect(x0+1, y0+1, x1-1, y1-1, mix(sf::Color::White, color, 0.5f));
      }

      levelBufWrite.fillRect(x0, y0, x1, y1, level);
    }

    return maxLevel;
  }

  void render() {


    int partitionWidth = buffer->getWidth() / config.numPartitionsX;
    int partitionHeight = buffer->getHeight() / config.numPartitionsY;

    if(wireframe) {
      buffer->clear(sf::Color::Black);
    }

#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < config.numPartitionsX * config.numPartitionsY; k++) {
      int i = k % config.numPartitionsX;
      int j = k / config.numPartitionsX;

      renderQuadrantRecursive(
          i * partitionWidth, j * partitionHeight,
          (i + 1) * partitionWidth,
          (j + 1) * partitionHeight, 100000.0, 0);
    }

    levelBufRead.swap(levelBufWrite);
  }

  void toggleWireframe() { wireframe = !wireframe; }
};
