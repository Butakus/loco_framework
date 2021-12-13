#ifndef LOCO_FRAMEWORK__UTILS_HPP_
#define LOCO_FRAMEWORK__UTILS_HPP_

#include <iterator>
#include <algorithm>
#include <chrono>

namespace loco
{

/* Find the index of the max element in a container */
template<typename ContainerType>
size_t find_max_index(const ContainerType& c)
{
    return std::distance(std::begin(c), std::max_element(std::begin(c), std::end(c)));
}

double tic()
{
    static std::chrono::time_point<std::chrono::steady_clock> last_tic__;
    auto new_tic = std::chrono::steady_clock::now();
    double micros = std::chrono::duration<double, std::micro>(new_tic - last_tic__).count();
    last_tic__ = new_tic;
    return micros;
}

/* Extract namespace from topic string */
std::string extract_ns(const std::string& topic)
{
    return topic.substr(1, topic.find('/', 1));
}


} // Namespace loco

#endif // LOCO_FRAMEWORK__UTILS_HPP_
