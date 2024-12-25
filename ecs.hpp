#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <bitset>
#include <queue>
#include <typeindex>

namespace ecs {

using ComponentTypeID = std::size_t;
using EntityID = std::size_t;

constexpr std::size_t MAX_COMPONENTS = 32;
using ComponentMask = std::bitset<MAX_COMPONENTS>;

// Component type ID generator
inline ComponentTypeID getComponentTypeID() {
    static ComponentTypeID lastID = 0;
    return lastID++;
}

template <typename T>
inline ComponentTypeID getComponentTypeID() noexcept {
    static ComponentTypeID typeID = getComponentTypeID();
    return typeID;
}

// IComponentArray: non-template base class for component arrays
class IComponentArray {
public:
    virtual ~IComponentArray() = default;
    virtual void entityDestroyed(EntityID entity) = 0;
};

// ComponentArray: stores actual component data
template<typename T>
class ComponentArray : public IComponentArray {
    std::vector<T> components;
    std::unordered_map<EntityID, size_t> entityToIndexMap;
    std::unordered_map<size_t, EntityID> indexToEntityMap;
    size_t size = 0;

public:
    void insertData(EntityID entity, T component) {
        size_t newIndex = size;
        entityToIndexMap[entity] = newIndex;
        indexToEntityMap[newIndex] = entity;
        if (newIndex >= components.size()) {
            components.push_back(std::move(component));
        } else {
            components[newIndex] = std::move(component);
        }
        ++size;
    }

    void removeData(EntityID entity) {
        if (entityToIndexMap.find(entity) == entityToIndexMap.end()) {
            return;
        }
        
        size_t indexOfRemovedEntity = entityToIndexMap[entity];
        size_t indexOfLastElement = size - 1;
        
        if (indexOfRemovedEntity != indexOfLastElement) {
            components[indexOfRemovedEntity] = std::move(components[indexOfLastElement]);
            EntityID entityOfLastElement = indexToEntityMap[indexOfLastElement];
            entityToIndexMap[entityOfLastElement] = indexOfRemovedEntity;
            indexToEntityMap[indexOfRemovedEntity] = entityOfLastElement;
        }

        entityToIndexMap.erase(entity);
        indexToEntityMap.erase(indexOfLastElement);
        --size;
    }

    T& getData(EntityID entity) {
        return components[entityToIndexMap[entity]];
    }

    const T& getData(EntityID entity) const {
        return components[entityToIndexMap[entity]];
    }

    bool hasEntity(EntityID entity) const {
        return entityToIndexMap.find(entity) != entityToIndexMap.end();
    }

    void entityDestroyed(EntityID entity) override {
        if (hasEntity(entity)) {
            removeData(entity);
        }
    }

    size_t getSize() const { return size; }
    
    auto begin() { return components.begin(); }
    auto end() { return components.begin() + size; }
    auto begin() const { return components.begin(); }
    auto end() const { return components.begin() + size; }
};

// ComponentManager: manages all component arrays
class ComponentManager {
    std::unordered_map<std::type_index, std::shared_ptr<IComponentArray>> componentArrays;
    std::unordered_map<std::type_index, ComponentTypeID> componentTypes;

public:
    template<typename T>
    void registerComponent() {
        auto typeName = std::type_index(typeid(T));
        componentTypes.insert({typeName, getComponentTypeID<T>()});
        componentArrays.insert({typeName, std::make_shared<ComponentArray<T>>()});
    }

    template<typename T>
    ComponentTypeID getComponentType() {
        auto typeName = std::type_index(typeid(T));
        return componentTypes[typeName];
    }

    template<typename T>
    void addComponent(EntityID entity, T component) {
        getComponentArray<T>()->insertData(entity, component);
    }

    template<typename T>
    void removeComponent(EntityID entity) {
        getComponentArray<T>()->removeData(entity);
    }

    template<typename T>
    T& getComponent(EntityID entity) {
        return getComponentArray<T>()->getData(entity);
    }

    template<typename T>
    const T& getComponent(EntityID entity) const {
        return getComponentArray<T>()->getData(entity);
    }

    template<typename T>
    bool hasComponent(EntityID entity) const {
        auto typeName = std::type_index(typeid(T));
        auto it = componentArrays.find(typeName);
        if (it == componentArrays.end()) return false;
        auto array = std::static_pointer_cast<ComponentArray<T>>(it->second);
        return array->hasEntity(entity);
    }

    template<typename T>
    std::shared_ptr<ComponentArray<T>> getComponentArray() {
        auto typeName = std::type_index(typeid(T));
        return std::static_pointer_cast<ComponentArray<T>>(componentArrays[typeName]);
    }

    void entityDestroyed(EntityID entity) {
        for (auto const& pair : componentArrays) {
            pair.second->entityDestroyed(entity);
        }
    }
};

// EntityManager: manages entity creation and destruction
class EntityManager {
    std::queue<EntityID> availableEntities;
    std::vector<ComponentMask> entityComponentMasks;
    uint32_t livingEntityCount = 0;
    static const size_t MAX_ENTITIES = 5000;

public:
    EntityManager() {
        for (EntityID entity = 0; entity < MAX_ENTITIES; ++entity) {
            availableEntities.push(entity);
        }
    }

    EntityID createEntity() {
        if (availableEntities.empty()) {
            throw std::runtime_error("Maximum number of entities exceeded");
        }

        EntityID id = availableEntities.front();
        availableEntities.pop();
        ++livingEntityCount;

        if (id >= entityComponentMasks.size()) {
            entityComponentMasks.resize(id + 1);
        }

        return id;
    }

    void destroyEntity(EntityID entity) {
        if (entity >= entityComponentMasks.size()) return;
        
        entityComponentMasks[entity].reset();
        availableEntities.push(entity);
        --livingEntityCount;
    }

    void setComponentMask(EntityID entity, ComponentMask mask) {
        entityComponentMasks[entity] = mask;
    }

    ComponentMask getComponentMask(EntityID entity) const {
        return entityComponentMasks[entity];
    }

    bool isAlive(EntityID entity) const {
        return entity < entityComponentMasks.size() && entityComponentMasks[entity].any();
    }

    size_t getLivingEntityCount() const { return livingEntityCount; }
    size_t getHighestEntity() const { return entityComponentMasks.size(); }
};

// System base class
class System {
protected:
    ComponentMask componentMask;

public:
    virtual ~System() = default;
    virtual void update(float dt) = 0;
    
    void setComponentMask(ComponentMask mask) {
        componentMask = mask;
    }
    
    const ComponentMask& getComponentMask() const {
        return componentMask;
    }

    template<typename T>
    void requireComponent() {
        componentMask.set(getComponentTypeID<T>());
    }
};

// World: main ECS manager
class World {
    std::unique_ptr<ComponentManager> componentManager;
    std::unique_ptr<EntityManager> entityManager;
    std::vector<std::unique_ptr<System>> systems;

public:
    World() 
        : componentManager(std::make_unique<ComponentManager>())
        , entityManager(std::make_unique<EntityManager>()) {}

    template<typename T>
    void registerComponent() {
        componentManager->registerComponent<T>();
    }

    template<typename T, typename... Args>
    T* registerSystem(Args&&... args) {
        auto system = std::make_unique<T>(std::forward<Args>(args)...);
        T* systemPtr = system.get();
        systems.push_back(std::move(system));
        return systemPtr;
    }

    void updateSystems(float dt) {
        for (auto& system : systems) {
            system->update(dt);
        }
    }

    EntityID createEntity() {
        return entityManager->createEntity();
    }

    void destroyEntity(EntityID entity) {
        componentManager->entityDestroyed(entity);
        entityManager->destroyEntity(entity);
    }

    bool isAlive(EntityID entity) const {
        return entityManager->isAlive(entity);
    }

    template<typename T>
    void addComponent(EntityID entity, T component) {
        componentManager->addComponent(entity, component);

        auto mask = entityManager->getComponentMask(entity);
        mask.set(componentManager->getComponentType<T>());
        entityManager->setComponentMask(entity, mask);
    }

    template<typename T>
    void removeComponent(EntityID entity) {
        componentManager->removeComponent<T>(entity);

        auto mask = entityManager->getComponentMask(entity);
        mask.set(componentManager->getComponentType<T>(), false);
        entityManager->setComponentMask(entity, mask);
    }

    template<typename T>
    T& getComponent(EntityID entity) {
        return componentManager->getComponent<T>(entity);
    }

    template<typename T>
    const T& getComponent(EntityID entity) const {
        return componentManager->getComponent<T>(entity);
    }

    template<typename T>
    bool hasComponent(EntityID entity) const {
        return componentManager->hasComponent<T>(entity);
    }

    template<typename T>
    ComponentArray<T>& getComponents() {
        return *componentManager->getComponentArray<T>();
    }

    template<typename... Components>
    std::vector<EntityID> getEntitiesWith() {
        ComponentMask mask;
        (mask.set(componentManager->getComponentType<Components>()), ...);

        std::vector<EntityID> entities;
        for (EntityID entity = 0; entity < entityManager->getHighestEntity(); ++entity) {
            if (entityManager->isAlive(entity) && 
                (entityManager->getComponentMask(entity) & mask) == mask) {
                entities.push_back(entity);
            }
        }
        return entities;
    }

    size_t getLivingEntityCount() const {
        return entityManager->getLivingEntityCount();
    }
};

// View: Helper for iterating entities with specific components
template<typename... Components>
class View {
    World& world;
    std::vector<EntityID> entities;

public:
    View(World& world) : world(world) {
        entities = world.getEntitiesWith<Components...>();
    }

    class Iterator {
        World& world;
        const std::vector<EntityID>& entities;
        size_t index;

    public:
        Iterator(World& world, const std::vector<EntityID>& entities, size_t index)
            : world(world), entities(entities), index(index) {}

        bool operator!=(const Iterator& other) const {
            return index != other.index;
        }

        Iterator& operator++() {
            ++index;
            return *this;
        }

        struct EntityComponents {
            EntityID entity;
            std::tuple<Components&...> components;
        };

        EntityComponents operator*() {
            return {
                entities[index],
                std::tuple<Components&...>(world.getComponent<Components>(entities[index])...)
            };
        }

        EntityID getEntity() const {
            return entities[index];
        }
    };

    Iterator begin() { return Iterator(world, entities, 0); }
    Iterator end() { return Iterator(world, entities, entities.size()); }
    size_t size() const { return entities.size(); }
};

} // namespace ecs