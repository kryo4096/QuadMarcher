#pragma once
#include <array>
#include <cstdint>

#include <SFML/Graphics.hpp>

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

inline sf::Color mix(const sf::Color &a, const sf::Color &b, float t) {
  return sf::Color(a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t,
                   a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t);
}
