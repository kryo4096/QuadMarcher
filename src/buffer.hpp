#pragma once

#include <SFML/Graphics.hpp>
#include <cassert>


template <typename T> class PixelBuffer {
protected:
  T *data;
  int width;
  int height;

public:
  PixelBuffer(int w, int h) : width(w), height(h) {
    data = new T[w * h];
  }

  PixelBuffer(const sf::Window &window)
      : PixelBuffer(window.getSize().x, window.getSize().y) {}

  void setPixel(int x, int y, T t) {
    if (x < 0 || y < 0)
      return;

    if (x >= width || y >= height)
      return;

    data[y * width + x] = t;
  }

  T getPixel(int x, int y) { return data[y * width + x]; }

  T getPixelOrDefault(int x, int y, T def = 0) {

    if (x < 0 || y < 0)
      return def;

    if (x >= width || y >= height)
      return def;

    return data[y * width + x];
  }

  void fillRect(int x0, int y0, int x1, int y1, T value) {
    for (int y = y0; y < y1; y++) {
      for (int x = x0; x < x1; x++) {
        data[y * width + x] = value;
      }
    }
  }

  void swap(PixelBuffer &other) {
    assert(width == other.width && height == other.height);
    T *temp = data;
    data = other.data;
    other.data = temp;
  }

  void clear(T value) {
    for (int i = 0; i < width * height; i++) {
      data[i] = value;
    }
  }

  int getWidth() const { return width; }
  int getHeight() const { return height; }
};


class CPUFramebuffer {
private:
  sf::Texture texture;
  sf::Uint8 *pixels;
  sf::Sprite sprite;
  int width;
  int height;

  sf::Uint8 *backbuffer;

public:
  CPUFramebuffer(int w, int h) : width(w), height(h) {
    // Allocate pixel buffer - 4 bytes per pixel (RGBA)
    pixels = new sf::Uint8[w * h * 4];
    backbuffer = new sf::Uint8[w * h * 4];

    // Create and configure texture
    texture.create(w, h);

    // Associate texture with sprite
    sprite.setTexture(texture);

    // Initialize to black
    clear(sf::Color::Black);
  }

  CPUFramebuffer(const sf::Window &window)
      : CPUFramebuffer(window.getSize().x, window.getSize().y) {}

  ~CPUFramebuffer() {
    delete[] pixels;
    delete[] backbuffer;
  }

  void clear(sf::Color color) {
    for (int i = 0; i < width * height * 4; i += 4) {
      pixels[i] = color.r;
      pixels[i + 1] = color.g;
      pixels[i + 2] = color.b;
      pixels[i + 3] = color.a;
    }
    update();
  }

  void setPixel(int x, int y, sf::Color color) {
    if (x >= width || y >= height)
      return;

    int base = (y * width + x) * 4;
    pixels[base] = color.r;
    pixels[base + 1] = color.g;
    pixels[base + 2] = color.b;
    pixels[base + 3] = color.a;
  }

  void mixPixel(int x, int y, sf::Color color, float t) {
    if (x >= width || y >= height)
      return;

    int base = (y * width + x) * 4;
    pixels[base] = pixels[base] + (color.r - pixels[base]) * t;
    pixels[base + 1] = pixels[base + 1] + (color.g - pixels[base + 1]) * t;
    pixels[base + 2] = pixels[base + 2] + (color.b - pixels[base + 2]) * t;
    pixels[base + 3] = pixels[base + 3] + (color.a - pixels[base + 3]) * t;
  }

  sf::Color getPixel(int x, int y) const {
    if (x >= width || y >= height)
      return sf::Color::Black;

    int base = (y * width + x) * 4;
    return sf::Color(backbuffer[base],backbuffer[base + 1], backbuffer[base + 2],
                     backbuffer[base + 3]);
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

  void fillRect(int x0, int y0, int x1, int y1, sf::Color color) {
    for (int y = y0; y < y1; y++) {
      for (int x = x0; x < x1; x++) {
        setPixel(x, y, color);
      }
    }
  }

  void mixRect(int x0, int y0, int x1, int y1, sf::Color color, float t) {

    x0 = std::max(0, x0);
    y0 = std::max(0, y0);

    x1 = std::min(static_cast<int>(width), x1);
    y1 = std::min(static_cast<int>(height), y1);

    for (int y = y0; y < y1; y++) {
      for (int x = x0; x < x1; x++) {
        mixPixel(x, y, color, t);
      }
    }
  }

  const sf::Texture &getTexture() const { return texture; }

  void swap() {
    sf::Uint8 *temp = pixels;
    pixels = backbuffer;
    backbuffer = temp;
  }

  void update() { texture.update(pixels); }

  // Add getters for dimensions if needed
  int getWidth() const { return width; }
  int getHeight() const { return height; }

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