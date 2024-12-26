#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Mouse.hpp>
#include <SFML/Window/Window.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>

const int BUFFER_WIDTH = 1024;
const int BUFFER_HEIGHT = 512;

class CPUFramebuffer {
private:
  sf::Texture texture;
  sf::Uint8 *pixels;
  sf::Sprite sprite;
  unsigned int width;
  unsigned int height;

public:
  CPUFramebuffer(unsigned int w, unsigned int h) : width(w), height(h) {
    // Allocate pixel buffer - 4 bytes per pixel (RGBA)
    pixels = new sf::Uint8[w * h * 4];

    // Create and configure texture
    texture.create(w, h);

    // Associate texture with sprite
    sprite.setTexture(texture);

    // Initialize to black
    clear(sf::Color::Black);
  }

  CPUFramebuffer(const sf::Window &window)
      : CPUFramebuffer(window.getSize().x, window.getSize().y) {}

  ~CPUFramebuffer() { delete[] pixels; }

  void clear(sf::Color color) {
    for (unsigned int i = 0; i < width * height * 4; i += 4) {
      pixels[i] = color.r;
      pixels[i + 1] = color.g;
      pixels[i + 2] = color.b;
      pixels[i + 3] = color.a;
    }
    update();
  }

  void setPixel(unsigned int x, unsigned int y, sf::Color color) {
    if (x >= width || y >= height)
      return;

    unsigned int base = (y * width + x) * 4;
    pixels[base] = color.r;
    pixels[base + 1] = color.g;
    pixels[base + 2] = color.b;
    pixels[base + 3] = color.a;
  }

  void mixPixel(unsigned int x, unsigned int y, sf::Color color, float t) {
    if (x >= width || y >= height)
      return;

    unsigned int base = (y * width + x) * 4;
    pixels[base] = pixels[base] + (color.r - pixels[base]) * t;
    pixels[base + 1] = pixels[base + 1] + (color.g - pixels[base + 1]) * t;
    pixels[base + 2] = pixels[base + 2] + (color.b - pixels[base + 2]) * t;
    pixels[base + 3] = pixels[base + 3] + (color.a - pixels[base + 3]) * t;
  }

  sf::Color getPixel(unsigned int x, unsigned int y) const {
    if (x >= width || y >= height)
      return sf::Color::Black;

    unsigned int base = (y * width + x) * 4;
    return sf::Color(pixels[base], pixels[base + 1], pixels[base + 2],
                     pixels[base + 3]);
  }

  void drawLine(int x1, int y1, int x2, int y2, sf::Color color) {
    // Bresenham's line algorithm
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
      setPixel(x1, y1, color);

      if (x1 == x2 && y1 == y2)
        break;

      int e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x1 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y1 += sy;
      }
    }
  }

  const sf::Texture &getTexture() const { return texture; }

  void update() { texture.update(pixels); }

  // Add getters for dimensions if needed
  unsigned int getWidth() const { return width; }
  unsigned int getHeight() const { return height; }

  void drawTo(sf::RenderWindow &window) {
    update();

    // Calculate scaling to fit window while maintaining aspect ratio
    sf::Vector2u windowSize = window.getSize();
    float scaleX = static_cast<float>(windowSize.x) / width;
    float scaleY = static_cast<float>(windowSize.y) / height;
    float scale = std::min(scaleX, scaleY);

    sprite.setScale(scale, scale);

    // Center the framebuffer in the window
    float scaledWidth = width * scale;
    float scaledHeight = height * scale;
    sprite.setPosition((windowSize.x - scaledWidth) / 2.0f,
                       (windowSize.y - scaledHeight) / 2.0f);

    window.draw(sprite);
  }
};
struct vec3 {
  float x, y, z;

  vec3() : x(0), y(0), z(0) {}
  vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

  vec3 &operator+=(const vec3 &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  vec3 &operator-=(const vec3 &rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  vec3 &operator*=(float rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    return *this;
  }

  vec3 operator+(const vec3 &rhs) const { return vec3(*this) += rhs; }
  vec3 operator-(const vec3 &rhs) const { return vec3(*this) -= rhs; }
  vec3 operator*(float rhs) const { return vec3(*this) *= rhs; }
  vec3 operator/(float rhs) const { return *this * (1.0f / rhs); }

  float dot(const vec3 &rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }

  vec3 cross(const vec3 &rhs) const {
    return {y * rhs.z - z * rhs.y, z * rhs.x - x * rhs.z,
            x * rhs.y - y * rhs.x};
  }

  float lengthSq() const { return dot(*this); }

  float length() const { return sqrtf(lengthSq()); }

  vec3 normalized() const {
    float len = length();
    return len > 0 ? *this * (1.0f / len) : vec3();
  }
};

class DepthBuffer {
private:
  float *depth;
  unsigned int width;
  unsigned int height;

public:
  DepthBuffer(unsigned int w, unsigned int h) : width(w), height(h) {
    depth = new float[w * h];
    clear();
  }

  DepthBuffer(const sf::Window &window)
      : DepthBuffer(window.getSize().x, window.getSize().y) {}

  ~DepthBuffer() { delete[] depth; }

  void clear() {
    for (unsigned int i = 0; i < width * height; i++) {
      depth[i] = 100.0f; // Initialize to maximum depth
    }
  }

  void setDepth(unsigned int x, unsigned int y, float d) {
    if (x >= width || y >= height)
      return;

    depth[y * width + x] = d;
  }

  float getDepth(unsigned int x, unsigned int y) const {
    if (x >= width || y >= height)
      return 1.0f;

    return depth[y * width + x];
  }

  bool testAndSet(unsigned int x, unsigned int y, float d) const {
    if (x >= width || y >= height)
      return false;

    if (d < depth[y * width + x]) {
      depth[y * width + x] = d;
      return true;
    } else {
      return false;
    }
  }
};

// Represents a normalized quaternion for 3D rotation
struct rotation {
  float x, y, z, w;

  rotation() : x(0), y(0), z(0), w(1) {} // Identity rotation
  rotation(float x_, float y_, float z_, float w_)
      : x(x_), y(y_), z(z_), w(w_) {}

  // Creates a rotation from axis and angle (in radians)
  static rotation fromAxisAngle(const vec3 &axis, float angle) {
    float halfAngle = angle * 0.5f;
    float s = sinf(halfAngle);
    vec3 normalizedAxis = axis.normalized();
    return {normalizedAxis.x * s, normalizedAxis.y * s, normalizedAxis.z * s,
            cosf(halfAngle)};
  }

  // Combines two rotations (this * rhs)
  rotation operator*(const rotation &rhs) const {
    return {w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
            w * rhs.y - x * rhs.z + y * rhs.w + z * rhs.x,
            w * rhs.z + x * rhs.y - y * rhs.x + z * rhs.w,
            w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z};
  }

  // Returns the inverse rotation
  rotation inverse() const {
    return {-x, -y, -z, w}; // Simplified because we maintain normalization
  }

  // Rotates a vector by this rotation
  vec3 operator*(const vec3 &v) const {
    // Optimized version of v' = q * v * q^-1 for unit quaternions
    vec3 u = {x, y, z};
    float s = w;

    vec3 result = u.cross(v) * 2.0f;
    return v + result * s + u.cross(result);
  }

  // Linear interpolation between rotations
  // Note: This is simpler than slerp but less accurate for large angles
  rotation lerp(const rotation &rhs, float t) const {
    // Prefer shortest path
    float dot = x * rhs.x + y * rhs.y + z * rhs.z + w * rhs.w;
    float sign = dot >= 0 ? 1.0f : -1.0f;

    rotation result = {x + (rhs.x * sign - x) * t, y + (rhs.y * sign - y) * t,
                       z + (rhs.z * sign - z) * t, w + (rhs.w * sign - w) * t};

    // Normalize to maintain unit quaternion
    float invLen = 1.0f / sqrtf(result.x * result.x + result.y * result.y +
                                result.z * result.z + result.w * result.w);

    result.x *= invLen;
    result.y *= invLen;
    result.z *= invLen;
    result.w *= invLen;

    return result;
  }

  rotation normalized() const {
    float invLen = 1.0f / sqrtf(x * x + y * y + z * z + w * w);
    return {x * invLen, y * invLen, z * invLen, w * invLen};
  }
};

struct rigidbody {
  vec3 position;
  rotation orientation;
  vec3 angular_momentum;
  vec3 linear_momentum;
  vec3 force;
  vec3 torque;
  float linear_friction = 0.0f;

  rigidbody()
      : position(0, 0, 0), orientation(), angular_momentum(0, 0, 0),
        linear_momentum(0, 0, 0) {}

  void update(float dt) {
    // Integrate linear momentum
    position += linear_momentum * dt;

    // Integrate angular momentum
    orientation = (orientation * rotation::fromAxisAngle(angular_momentum, dt))
                      .normalized();

    linear_momentum += force * dt;
    angular_momentum += torque * dt;

    // Dampen angular momentum
    angular_momentum *= 0.99f;
    linear_momentum *= 1.0f / (1.0f + linear_friction * dt);
  }

  void addForce(vec3 _force) { force += _force; }

  void addTorque(vec3 _torque) { torque += _torque; }

  vec3 translate(vec3 bodyPos) { return orientation * bodyPos + position; }
};

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

template <size_t N> class ColorMap {
  std::array<sf::Color, N> colors;
  std::array<float, N> positions; // Normalized positions [0,1]

public:
  ColorMap(const std::array<sf::Color, N> &cols,
           const std::array<float, N> &pos)
      : colors(cols), positions(pos) {}

  sf::Color operator()(float t) const {
    // Clamp input to [0,1]
    t = std::min(1.0f, std::max(0.0f, t));

    // Find the segment
    size_t i = 0;
    while (i < N - 1 && positions[i + 1] < t)
      i++;

    // Handle edge cases
    if (i >= N - 1)
      return colors[N - 1];
    if (i == 0 && t < positions[0])
      return colors[0];

    // Linear interpolation between colors
    float segment_t = (t - positions[i]) / (positions[i + 1] - positions[i]);

    return sf::Color(
        static_cast<uint8_t>(colors[i].r +
                             segment_t * (colors[i + 1].r - colors[i].r)),
        static_cast<uint8_t>(colors[i].g +
                             segment_t * (colors[i + 1].g - colors[i].g)),
        static_cast<uint8_t>(colors[i].b +
                             segment_t * (colors[i + 1].b - colors[i].b)),
        static_cast<uint8_t>(colors[i].a +
                             segment_t * (colors[i + 1].a - colors[i].a)));
  }
};

// Common colormap definitions
namespace colormaps {
// Grayscale: Black to White
static const auto grayscale =
    ColorMap<2>({sf::Color::Black, sf::Color::White}, {0.0f, 1.0f});

// Red to Blue (diverging)
static const auto redblue = ColorMap<3>(
    {
        sf::Color(220, 50, 50),   // Red
        sf::Color(240, 240, 240), // White
        sf::Color(50, 50, 220)    // Blue
    },
    {0.0f, 0.5f, 1.0f});

// Jet (like MATLAB's jet colormap)
static const auto jet = ColorMap<5>(
    {
        sf::Color(0, 0, 140),   // Dark blue
        sf::Color(0, 255, 255), // Cyan
        sf::Color(0, 255, 0),   // Green
        sf::Color(255, 255, 0), // Yellow
        sf::Color(255, 0, 0)    // Red
    },
    {0.0f, 0.25f, 0.5f, 0.75f, 1.0f});

// Turbo (Google's improved rainbow colormap)
static const auto turbo = ColorMap<7>(
    {
        sf::Color(48, 18, 59),   // Dark purple
        sf::Color(86, 91, 184),  // Blue
        sf::Color(36, 162, 217), // Light blue
        sf::Color(31, 201, 163), // Cyan
        sf::Color(122, 209, 81), // Green
        sf::Color(249, 189, 38), // Yellow
        sf::Color(240, 39, 35)   // Red
    },
    {0.0f, 0.17f, 0.33f, 0.5f, 0.67f, 0.83f, 1.0f});

// Viridis (perceptually uniform)
static const auto viridis = ColorMap<5>(
    {
        sf::Color(68, 1, 84),    // Dark purple
        sf::Color(58, 82, 139),  // Blue
        sf::Color(32, 144, 140), // Teal
        sf::Color(94, 201, 97),  // Green
        sf::Color(253, 231, 36)  // Yellow
    },
    {0.0f, 0.25f, 0.5f, 0.75f, 1.0f});

// Heat (Black-Red-Yellow-White)
static const auto heat =
    ColorMap<4>({sf::Color::Black, sf::Color(255, 0, 0), // Red
                 sf::Color(255, 255, 0),                 // Yellow
                 sf::Color::White},
                {0.0f, 0.33f, 0.66f, 1.0f});
} // namespace colormaps

sf::Color mix(const sf::Color &a, const sf::Color &b, float t) {
  return sf::Color(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t,
                   a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t);
}

struct FirstPersonMouse {
  sf::Window *window;
  float sensitivity = 0.1f;
  float yaw = 0;
  float pitch = 0;

  FirstPersonMouse(sf::Window *window) : window(window) {

    window->setMouseCursorVisible(false);
  }

  void update() {
    sf::Vector2i center(window->getPosition().x + window->getSize().x / 2,
                        window->getPosition().y + window->getSize().y / 2);

    sf::Vector2i mouse = sf::Mouse::getPosition();

    yaw += (mouse.x - center.x) * sensitivity;
    pitch += (mouse.y - center.y) * sensitivity;

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

  FirstPersonController(sf::Window* window) : cam(60.0f, 16.0f / 9.0f, 0.1f, 100.0f), mouse(window) {
    cam.position = {0, 0, -5};
    cam.lookAt({0, 0, 0});
  }

  void update(float dt) {

    const float speed = 5.0f;

    vec3 velocity = {0, 0, 0};

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
      velocity.z += 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
      velocity.z -= 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
      velocity.x += 1;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
      velocity.x -= 1;
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

    //rb.orientation = cam.orientation;
  }
};

class RayMarcher {
public:
  CPUFramebuffer *buffer;
  DepthBuffer *depthBuffer;
  Camera *cam;

  RayMarcher(CPUFramebuffer *buffer, DepthBuffer *depthBuffer, Camera *cam)
      : buffer(buffer), depthBuffer(depthBuffer), cam(cam) {}

  float sphere(const vec3 &p, float radius) {
    return p.length() - radius;
  }

  float torus(const vec3 &p, float majorRadius, float minorRadius) {
    return sqrtf(p.x * p.x + p.y * p.y) - majorRadius;
  }

  float plane(const vec3 &p, const vec3 &n, float d) {
    return p.dot(n) + d;
  }

  float scene(const vec3 &p) {
    return std::min(sphere(p, 1.0f), torus(p, 1.0f, 0.5f));
  }

  float raymarch(const vec3 &origin, const vec3 &direction, float maxDistance,
                 float epsilon, float minDistance) {
    float t = minDistance;

    for (int i = 0; i < 100; i++) {
      vec3 p = origin + direction * t;
      float dist = scene(p);

      if (dist < epsilon || t > maxDistance)
        break;

      t += dist;
    }

    return t;
  }

  void render() {
    for (int x = 0; x < buffer->getWidth(); x++) {
      for (int y = 0; y < buffer->getHeight(); y++) {
        vec3 screenPos = {2.0f * x / buffer->getWidth() - 1.0f,
                          1.0f - 2.0f * y / buffer->getHeight(), 0};

        vec3 worldPos = cam->screenToWindow(screenPos, buffer->getWidth(),
                                            buffer->getHeight());

        vec3 direction = (worldPos - cam->position).normalized();

        float depth = raymarch(cam->position, direction, cam->farPlane, 0.01f,
                               0.01f);
          sf::Color color = colormaps::grayscale(exp(-depth * 0.1f));
          buffer->setPixel(x, y, color);
      }
    }
  }
};

int main() {
  sf::RenderWindow window(sf::VideoMode(1024, 512), "Raymarching Test");
  window.setFramerateLimit(60);

  CPUFramebuffer framebuffer(window);
  DepthBuffer depthBuffer(window);

  int mouse_x = 0;
  int mouse_y = 0;

  FirstPersonController player(&window);

  float t = 0;

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::mt19937 rng;

  rigidbody rb;

  RayMarcher raymarcher(&framebuffer, &depthBuffer, &player.cam);

  //rb.addTorque({0.0, 1.0, 0.0});
  //rb.addTorque({0.0, 0.0, 1.0});
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
      }
    }

    for (int x = 0; x < BUFFER_WIDTH; x++) {
      for (int y = 0; y < BUFFER_HEIGHT; y++) {

        framebuffer.mixPixel(x, y, sf::Color::Black, 0.5f);
      }
    }

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    for (int i = 0; i < 100000; i++) {

      float u = dist(rng) * 1.0f;
      float v = dist(rng);
      // float w = dist(rng);

      // simple torus
      vec3 p = {cosf(2.0f * M_PI * u) * (1.0f + 0.5f * cosf(2.0f * M_PI * v)),
                sinf(2.0f * M_PI * u) * (1.0f + 0.5f * cosf(2.0f * M_PI * v)),
                0.5f * sinf(2.0f * M_PI * v)};

      vec3 n = {cosf(2.0f * M_PI * u) * cosf(2.0f * M_PI * v),
                sinf(2.0f * M_PI * u) * cosf(2.0f * M_PI * v),
                sinf(2.0f * M_PI * v)};

      n = rb.translate(n);

      vec3 sp = player.cam.worldToWindow(
          rb.translate(p), framebuffer.getWidth(), framebuffer.getHeight());

      sf::Color color = colormaps::grayscale(n.x * 0.5f + 0.5f);



      if (sp.z > 0.0f && depthBuffer.testAndSet(sp.x, sp.y, sp.z)) {
        framebuffer.mixPixel(sp.x, sp.y, color, 0.5f);
      }
    }

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::cout << "Time difference = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;

    depthBuffer.clear();

    framebuffer.drawTo(window);

    window.display();

    t += 0.01f;
    rb.update(0.01f);
    player.update(0.01f);
  }

  return 0;
}