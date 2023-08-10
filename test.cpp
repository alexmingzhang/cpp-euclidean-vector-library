#include <complex>
#include <iostream>

#include "euclidean_vector.hpp"

// Overloads the output stream operator to print a Euclidean vector.
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const EucVec<T, N> &vec) {
    os << '<';
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        if (it != vec.begin()) {
            os << ", ";
        }
        os << *it;
    }
    os << '>';

    return os;
}

int main() {
    // Real-valued vectors
    constexpr EucVec<double, 3> v1{1.5, 2.5, 3.5};
    constexpr EucVec<double, 3> v2{4.5, 5.5, 6.5};
    constexpr double s1 = 3.0;

    std::cout << v1 << " + " << v2 << " = " << v1 + v2 << '\n';
    std::cout << v1 << " - " << v2 << " = " << v1 - v2 << '\n';
    std::cout << v1 << " x " << v2 << " = " << v1.cross(v2) << '\n';
    std::cout << v1 << " . " << v2 << " = " << v1.dot(v2) << '\n';
    std::cout << v1 << " * " << s1 << " = " << v1 * s1 << '\n';
    std::cout << "Euclidean norm of " << v1 << ": " << v1.norm() << '\n';
    std::cout << "3-norm of " << v1 << ": " << v1.norm<3>() << '\n';
    std::cout << "Normalized vector " << v1 << ": " << v1.normalize<3>()
              << '\n';
    std::cout << "Distance between " << v1 << " and " << v2 << ": "
              << v1.dist_to(v2) << '\n';
    std::cout << "Angle between " << v1 << " and " << v2 << ": "
              << v1.radians_between(v2) << " radians" << '\n';
    std::cout << "Projection of " << v1 << " onto " << v2 << ": "
              << v1.project_onto(v2) << '\n';

    // Complex-valued vectors
    using namespace std::complex_literals;
    constexpr EucVec<std::complex<double>, 3> v3{
        std::complex<double>(1.0, 1.0), std::complex<double>(2.0, 2.0),
        std::complex<double>(3.0, 3.0)};
    constexpr EucVec<std::complex<double>, 3> v4{
        std::complex<double>(4.0, 4.0), std::complex<double>(5.0, 5.0),
        std::complex<double>(6.0, 6.0)};
    constexpr std::complex<double> s2{3, 3};

    std::cout << v3 << " + " << v4 << " = " << v3 + v4 << '\n';
    std::cout << v3 << " - " << v4 << " = " << v3 - v4 << '\n';
    std::cout << v3 << " x " << v4 << " = " << v3.cross(v4) << '\n';
    std::cout << v3 << " . " << v4 << " = " << v3.dot(v4) << '\n';
    std::cout << v3 << " * " << s2 << " = " << v3 * s2 << '\n';
    std::cout << "Euclidean norm of " << v3 << ": " << v3.norm() << '\n';

    return 0;
}
