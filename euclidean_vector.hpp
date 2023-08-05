#ifndef EUCLIDEAN_VECTOR_HPP
#define EUCLIDEAN_VECTOR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <concepts>
#include <initializer_list>
#include <numeric>

// clang-format off
template <typename T>
concept AcceptableScalar = 
    std::equality_comparable<T> &&
    requires(T a, T b) {
        { a + b } -> std::same_as<T>; // Closure of addition
        { a - b } -> std::same_as<T>; // Closure of subtraction
        { a * b } -> std::same_as<T>; // Closure of multiplication
        { a / b } -> std::same_as<T>; // Closure of division
    };
// clang-format on

template <AcceptableScalar T, std::size_t N>
class EucVec {
public:
    // Member Types
    using container = std::array<T, N>;
    using value_type = T;
    using size_type = typename container::size_type;
    using difference_type = typename container::difference_type;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using iterator = typename container::iterator;
    using const_iterator = typename container::const_iterator;
    using reverse_iterator = typename container::reverse_iterator;
    using const_reverse_iterator = typename container::const_reverse_iterator;

    // Constructors
    constexpr EucVec() noexcept = default;
    constexpr EucVec(const EucVec &) = default;
    constexpr EucVec(EucVec &&) noexcept = default;
    constexpr EucVec(std::initializer_list<T> init) {
        std::copy(init.begin(), init.end(), m_data.begin());
    };

    // Destructor
    constexpr ~EucVec() = default;

    // Assignment
    constexpr EucVec &operator=(const EucVec &) = default;

    // Comparison
    constexpr bool operator==(const EucVec &) const = default;
    constexpr bool operator!=(const EucVec &) const = default;

    // Element Access
    constexpr reference at(size_type pos) { return m_data.at(pos); }
    constexpr const_reference at(size_type pos) const { return m_data.at(pos); }
    constexpr reference operator[](size_type pos) { return m_data[pos]; }
    constexpr const_reference operator[](size_type pos) const {
        return m_data[pos];
    }
    constexpr reference front() { return m_data.front(); }
    constexpr const_reference front() const { return m_data.front(); }
    constexpr reference back() { return m_data.back(); }
    constexpr const_reference back() const { return m_data.back(); }
    constexpr T *data() noexcept { return m_data.data(); }
    constexpr const T *data() const noexcept { return m_data.data(); }

    // Iterators
    constexpr iterator begin() noexcept { return m_data.begin(); }
    constexpr const_iterator begin() const noexcept { return m_data.begin(); }
    constexpr const_iterator cbegin() const noexcept { return m_data.cbegin(); }
    constexpr iterator end() noexcept { return m_data.end(); }
    constexpr const_iterator end() const noexcept { return m_data.end(); }
    constexpr const_iterator cend() const noexcept { return m_data.cend(); }
    constexpr reverse_iterator rbegin() noexcept { return m_data.rbegin(); }
    constexpr const_reverse_iterator rbegin() const noexcept {
        return m_data.rbegin();
    }
    constexpr const_reverse_iterator crbegin() const noexcept {
        return m_data.crbegin();
    }
    constexpr reverse_iterator rend() noexcept { return m_data.rend(); }
    constexpr const_reverse_iterator rend() const noexcept {
        return m_data.rend();
    }
    constexpr const_reverse_iterator crend() const noexcept {
        return m_data.crend();
    }

    // Capacity
    [[nodiscard]] constexpr bool empty() const noexcept {
        return m_data.empty();
    }
    [[nodiscard]] constexpr size_type size() const noexcept {
        return m_data.size();
    }
    [[nodiscard]] constexpr size_type max_size() const noexcept {
        return m_data.max_size();
    }

    // Modifiers
    constexpr void fill(const T &value) { m_data.fill(value); }
    constexpr void swap(EucVec &other) noexcept { m_data.swap(other.m_data); }

    // Operations
    [[nodiscard]] constexpr T dot(const EucVec &) const;
    [[nodiscard]] constexpr T norm() const;
    [[nodiscard]] constexpr T norm_sqr() const;
    [[nodiscard]] constexpr EucVec normalize() const;
    [[nodiscard]] constexpr T dist_to(const EucVec &) const;
    [[nodiscard]] constexpr T is_orthogonal_to(const EucVec &) const;
    [[nodiscard]] constexpr T angle_between(const EucVec &) const;
    [[nodiscard]] constexpr EucVec project_onto(const EucVec &) const;
    [[nodiscard]] constexpr EucVec cross(const EucVec &) const
        requires(N == 3);

private:
    container m_data;
};

template <AcceptableScalar T, std::size_t N>
constexpr T EucVec<T, N>::dot(const EucVec<T, N> &other) const {
    return std::transform_reduce(begin(), end(), other.begin(),
                                 static_cast<T>(0));
}

template <AcceptableScalar T, std::size_t N>
constexpr T EucVec<T, N>::norm() const {
    return std::sqrt(this->dot(*this));
}

template <AcceptableScalar T, std::size_t N>
constexpr T EucVec<T, N>::norm_sqr() const {
    return this->dot(*this);
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> EucVec<T, N>::normalize() const {
    return *this / this->norm();
}

template <AcceptableScalar T, std::size_t N>
constexpr T EucVec<T, N>::dist_to(const EucVec<T, N> &other) const {
    return (*this - other).norm();
}

template <AcceptableScalar T, std::size_t N>
constexpr T EucVec<T, N>::is_orthogonal_to(const EucVec<T, N> &other) const {
    return this->dot(other) == 0;
}

template <AcceptableScalar T, std::size_t N>
constexpr T EucVec<T, N>::angle_between(const EucVec<T, N> &other) const {
    return std::acos(this->dot(other) / (this->norm() * other.norm()));
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> EucVec<T, N>::project_onto(
    const EucVec<T, N> &other) const {
    return (this->dot(other) / this->dot(*this)) * *this;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> EucVec<T, N>::cross(const EucVec<T, N> &other) const
    requires(N == 3)
{
    return EucVec<T, 3>({m_data[1] * other[2] - m_data[2] * other[1],
                         m_data[2] * other[0] - m_data[0] * other[2],
                         m_data[0] * other[1] - m_data[1] * other[0]});
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> operator+(const EucVec<T, N> &lhs,
                                 const EucVec<T, N> &rhs) {
    EucVec<T, N> sum;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), sum.begin(),
                   std::plus<T>());
    return sum;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> &operator+=(EucVec<T, N> &lhs, const EucVec<T, N> &rhs) {
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                   std::plus<T>());
    return lhs;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> operator-(const EucVec<T, N> &lhs,
                                 const EucVec<T, N> &rhs) {
    EucVec<T, N> diff;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), diff.begin(),
                   std::minus<T>());
    return diff;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> &operator-=(EucVec<T, N> &lhs, const EucVec<T, N> &rhs) {
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                   std::minus<T>());
    return lhs;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> operator*(const T &scalar, const EucVec<T, N> &vec) {
    EucVec<T, N> prod;
    std::transform(vec.begin(), vec.end(), prod.begin(),
                   [&scalar](const T &val) { return val * scalar; });
    return prod;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> operator*(const EucVec<T, N> &vec, const T &scalar) {
    return scalar * vec;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> &operator*=(EucVec<T, N> &vec, const T &scalar) {
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [&scalar](const T &val) { return val * scalar; });
    return vec;
}

// Unary negation
template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> operator-(const EucVec<T, N> &vec) {
    return -1 * vec;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> operator/(const EucVec<T, N> &vec, const T &scalar) {
    EucVec<T, N> quot;
    std::transform(vec.begin(), vec.end(), quot.begin(),
                   [&scalar](const T &val) { return val / scalar; });
    return quot;
}

template <AcceptableScalar T, std::size_t N>
constexpr EucVec<T, N> &operator/=(EucVec<T, N> &vec, const T &scalar) {
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [&scalar](const T &val) { return val / scalar; });
    return vec;
}

#endif  // #ifndef EUCLIDEAN_VECTOR_HPP
