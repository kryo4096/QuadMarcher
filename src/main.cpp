#include <SFML/Graphics.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Mouse.hpp>

const int BUFFER_WIDTH = 256;
const int BUFFER_HEIGHT = 256;

int main() {
  sf::RenderWindow window(sf::VideoMode(600, 600),
                          "Raymarching Test");
  window.setFramerateLimit(60);

  sf::Texture texture{};
  texture.create(BUFFER_WIDTH, BUFFER_HEIGHT);

  std::vector<sf::Uint8> pixels(BUFFER_WIDTH * BUFFER_HEIGHT * 4, 0);

  // set alpha to 255

  for (int i = 0; i < BUFFER_WIDTH * BUFFER_HEIGHT; i++) {
    pixels[i * 4 + 3] = 255;
  }

  int cc = 0;
  int kks = 0;

  std::vector<uint8_t> laplacian(BUFFER_WIDTH * BUFFER_HEIGHT * 4, 0);

  int mouse_x = 0;
  int mouse_y = 0;

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

        if (event.key.code == sf::Keyboard::Space) {
          kks += 1;
        }
      }

      if (event.type == sf::Event::MouseMoved) {
        mouse_x = event.mouseMove.x;
        mouse_y = event.mouseMove.y;
      }

      if (event.type == sf::Event::MouseButtonPressed) {
        cc = (cc + 1) % 3;
      }

      
    }

    int pixels_x = mouse_x / (600.0f / BUFFER_WIDTH);
    int pixels_y = mouse_y / (600.0f / BUFFER_HEIGHT);

    pixels[(pixels_x + pixels_y * BUFFER_WIDTH) * 4 + cc] = 255;


    

    for (int y = 0; y < BUFFER_HEIGHT; y++) {
      for (int x = 0; x < BUFFER_WIDTH; x++) {
        for (int c = 0; c < 3; c++) {
          int index = (x + y * BUFFER_WIDTH) * 4 + c;

          int y_top = (y - 1 + BUFFER_HEIGHT) % BUFFER_HEIGHT;
          int y_bottom = (y + 1) % BUFFER_HEIGHT;

          int x_left = (x - 1 + BUFFER_WIDTH) % BUFFER_WIDTH;
          int x_right = (x + 1) % BUFFER_WIDTH;


          int index_top = (x + y_top * BUFFER_WIDTH) * 4 + c;
          int index_bottom = (x + y_bottom * BUFFER_WIDTH) * 4 + c;
          int index_left = (x_left + y * BUFFER_WIDTH) * 4 + c;
          int index_right = (x_right + y * BUFFER_WIDTH) * 4 + c;
          int index_wplus = (x + y * BUFFER_WIDTH) * 4 + (c + 1) % 3;
          int index_wminus = (x + y * BUFFER_WIDTH) * 4 + (c - 1 + 3) % 3;
          
          int lap = 0;
          lap += pixels[index_top];
          lap += pixels[index_bottom];
          lap += pixels[index_left];
          lap += pixels[index_right];
          lap += pixels[index_wplus];
          lap += pixels[index_wminus];

          lap -= 6 * pixels[index];

          laplacian[index] = lap >> 1+kks+c;
        }
      }
    }

    for (int y = 0; y < BUFFER_HEIGHT; y++) {
      for (int x = 0; x < BUFFER_WIDTH; x++) {
        for (int c = 0; c < 3; c++) {
          int index = (x + y * BUFFER_WIDTH) * 4 + c;
          pixels[index] = pixels[index] + (laplacian[index] >> 1);
          pixels[index] -= (pixels[index] >> 2) + !pixels[index];
        }
      }
    }

    //  for (int y = 0; y < WINDOW_HEIGHT; y++) {
    //   for (int x = 0; x < WINDOW_WIDTH; x++) {
    //     for (int c = 0; c < 3; c++) {
    //       int index = (x + y * WINDOW_WIDTH) * 4 + c;
    //       pixels[index] = pixels[(index+1)%(WINDOW_WIDTH * WINDOW_HEIGHT * 4)];
    //     }
    //   }
    //  }



    texture.update(pixels.data());

    sf::Sprite sprite(texture, sf::IntRect(0, 0, 600, 600));

    sprite.setScale(600.0f / BUFFER_WIDTH, 600.0f / BUFFER_HEIGHT);

    window.draw(sprite);

    window.display();
  }

  return 0;
}