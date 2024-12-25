#include "ecs.hpp"
#include <SFML/Graphics.hpp>
#include <cmath>
#include <numeric>

// Components
struct Position { float x, y; };
struct Velocity { float dx, dy; };
struct Renderable { 
    sf::CircleShape shape;
    Renderable(float radius, sf::Color color) : shape(radius) {
        shape.setFillColor(color);
        shape.setOrigin(radius, radius);
    }
};
struct Item {
    enum class Type { Rock, Sword } type;
    bool isHeld = false;
};
struct Enemy {
    bool isAlive = true;
    float knockbackForce = 50.0f;
    float size;
};
struct Player {};

// Systems
class RenderSystem : public ecs::System {
    sf::RenderWindow& window;
    ecs::World& world;

public:
    RenderSystem(sf::RenderWindow& window, ecs::World& world) 
        : window(window), world(world) {
        requireComponent<Position>();
        requireComponent<Renderable>();
    }

    void update(float dt) override {
        window.clear(sf::Color::White);
        
        for (auto [pos, renderable] : ecs::View<Position, Renderable>(world)) {
            renderable.shape.setPosition(pos.x, pos.y);
            window.draw(renderable.shape);
        }
        
        window.display();
    }
};

class MovementSystem : public ecs::System {
    ecs::World& world;
public:
    MovementSystem(ecs::World& world) : world(world) {
        requireComponent<Position>();
        requireComponent<Velocity>();
    }

    void update(float dt) override {
        for (auto [pos, vel] : ecs::View<Position, Velocity>(world)) {
            pos.x = std::clamp(pos.x + vel.dx * dt, 0.0f, 800.0f);
            pos.y = std::clamp(pos.y + vel.dy * dt, 0.0f, 600.0f);
        }
    }
};

class PlayerControlSystem : public ecs::System {
    ecs::World& world;
    float playerSpeed = 300.0f;

public:
    PlayerControlSystem(ecs::World& world) : world(world) {
        requireComponent<Player>();
        requireComponent<Velocity>();
    }

    void update(float dt) override {
        for (auto [vel,_] : ecs::View<Velocity, Player>(world)) {
            vel = {0, 0};
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))  vel.dx = -playerSpeed;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) vel.dx = playerSpeed;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))    vel.dy = -playerSpeed;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))  vel.dy = playerSpeed;
        }
    }
};

class ItemSystem : public ecs::System {
    ecs::World& world;
public:
    ItemSystem(ecs::World& world) : world(world) {
        requireComponent<Item>();
        requireComponent<Position>();
    }

    void update(float dt) override {
        // Find player position
        Position playerPos;
        for (auto [pos,_] : ecs::View<Position, Player>(world)) {
            playerPos = pos;
            break;
        }

        // Handle pickup
        for (const auto& view : ecs::View<Position, Item>(world)) {
            auto [pos, item] = view;
            if (!item.isHeld) {
                float dx = pos.x - playerPos.x;
                float dy = pos.y - playerPos.y;
                if (std::sqrt(dx * dx + dy * dy) < 40) {
                    item.isHeld = true;
                }
            }
        }

        // Handle throwing
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
            for (const auto& view : ecs::View<Position, Velocity, Item>(world)) {
                auto [pos, vel, item] = view;
                if (!item.isHeld) continue;

                auto targetPos = findClosestEnemy(pos);
                if (targetPos.x >= 0) {
                    float dx = targetPos.x - pos.x;
                    float dy = targetPos.y - pos.y;
                    float len = std::sqrt(dx * dx + dy * dy);
                    if (len > 0) {
                        vel.dx = (dx / len) * 500;
                        vel.dy = (dy / len) * 500;
                    }
                }
                item.isHeld = false;
            }
        }
    }

private:
    Position findClosestEnemy(const Position& fromPos) {
        Position closest{-1, -1};
        float minDist = std::numeric_limits<float>::max();

        for (const auto& view : ecs::View<Position, Enemy>(world)) {
            auto [pos, enemy] = view;
            if (!enemy.isAlive) continue;

            float dx = pos.x - fromPos.x;
            float dy = pos.y - fromPos.y;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist < minDist) {
                minDist = dist;
                closest = pos;
            }
        }

        return closest;
    }
};

class EnemyAISystem : public ecs::System {
    ecs::World& world;
    float enemySpeed = 100.0f;

public:
    EnemyAISystem(ecs::World& world) : world(world) {
        requireComponent<Enemy>();
        requireComponent<Position>();
        requireComponent<Velocity>();
    }

    void update(float dt) override {
        Position playerPos;
        for (auto [pos, _] : ecs::View<Position, Player>(world)) {
            playerPos = pos;
            break;
        }

        for (auto [pos, vel, enemy] : ecs::View<Position, Velocity, Enemy>(world)) {
            if (!enemy.isAlive) continue;

            float dx = playerPos.x - pos.x;
            float dy = playerPos.y - pos.y;
            float len = std::sqrt(dx * dx + dy * dy);
            
            if (len > 0) {
                vel.dx = (dx / len) * enemySpeed;
                vel.dy = (dy / len) * enemySpeed;
            }
        }
    }
};

class CollisionSystem : public ecs::System {
    ecs::World& world;

public:
    CollisionSystem(ecs::World& world) : world(world) {
        requireComponent<Position>();
    }

    void update(float dt) override {
        for (const auto& enemyView : ecs::View<Position, Enemy>(world)) {
            auto [enemyPos, enemy] = enemyView;
            if (!enemy.isAlive) continue;

            for (const auto& itemView : ecs::View<Position, Item>(world)) {
                auto [itemPos, item] = itemView;
                if (item.isHeld) continue;

                float dx = itemPos.x - enemyPos.x;
                float dy = itemPos.y - enemyPos.y;
                
                if (std::sqrt(dx * dx + dy * dy) < enemy.size * 1.5f) {
                    handleCollision(enemyView.getEntity(), item.type, dx, dy);
                    world.destroyEntity(itemView.getEntity());
                }
            }
        }
    }

private:
    void handleCollision(ecs::EntityID enemyId, Item::Type itemType, float dx, float dy) {
        auto& enemy = world.getComponent<Enemy>(enemyId);
        if (itemType == Item::Type::Sword) {
            enemy.isAlive = false;
        } else {
            auto& vel = world.getComponent<Velocity>(enemyId);
            float len = std::sqrt(dx * dx + dy * dy);
            vel.dx = (dx / len) * enemy.knockbackForce;
            vel.dy = (dy / len) * enemy.knockbackForce;
        }
    }
};

class EnemySpawnSystem : public ecs::System {
    ecs::World& world;
    sf::Clock spawnClock;
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> pos_dist{0, 800};
    std::uniform_real_distribution<> size_dist{10, 40};

public:
    EnemySpawnSystem(ecs::World& world) : world(world) {}

    void update(float dt) override {
        if (spawnClock.getElapsedTime().asSeconds() > 5.0f) {
            spawnEnemy();
            spawnClock.restart();
        }
    }

    void spawnEnemy() {
        auto entity = world.createEntity();
        float size = size_dist(gen);

        world.addComponent(entity, Position{pos_dist(gen), pos_dist(gen)});
        world.addComponent(entity, Velocity{0, 0});
        world.addComponent(entity, Renderable(size, sf::Color::Red));
        world.addComponent(entity, Enemy{true, 50.0f, size});
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "ECS Game");
    window.setFramerateLimit(60);

    ecs::World world;

    // Register components
    world.registerComponent<Position>();
    world.registerComponent<Velocity>();
    world.registerComponent<Renderable>();
    world.registerComponent<Item>();
    world.registerComponent<Enemy>();
    world.registerComponent<Player>();

    // Create systems
    world.registerSystem<RenderSystem>(window, world);
    world.registerSystem<MovementSystem>(world);
    world.registerSystem<PlayerControlSystem>(world);
    world.registerSystem<ItemSystem>(world);
    world.registerSystem<EnemyAISystem>(world);
    world.registerSystem<CollisionSystem>(world);
    auto spawnSystem = world.registerSystem<EnemySpawnSystem>(world);

    // Create player
    auto player = world.createEntity();
    world.addComponent(player, Position{400, 300});
    world.addComponent(player, Velocity{0, 0});
    world.addComponent(player, Renderable(20, sf::Color::Blue));
    world.addComponent(player, Player{});

    // Create items
    auto createItem = [&](float x, float y, Item::Type type) {
        auto entity = world.createEntity();
        world.addComponent(entity, Position{x, y});
        world.addComponent(entity, Velocity{0, 0});
        world.addComponent(entity, Renderable(10, type == Item::Type::Rock ? sf::Color::Gray : sf::Color::Yellow));
        world.addComponent(entity, Item{type});
    };

    createItem(200, 200, Item::Type::Rock);
    createItem(600, 400, Item::Type::Sword);
    spawnSystem->spawnEnemy();

    sf::Clock clock;
    while (window.isOpen()) {
        float dt = clock.restart().asSeconds();

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        world.updateSystems(dt);
    }

    return 0;
}