/**
 * @file euclidean_vector.hpp
 * @author Alex Zhang (GitHub: alexmingzhang)
 * @brief Header-only C++20 library for Euclidean vectors in n-dimensions
 * @date 2023-08-09
 *
 */

#ifndef EUCLIDEAN_VECTOR_HPP
#define EUCLIDEAN_VECTOR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <numeric>

/// @defgroup concepts Concepts
/// @{
/**
 * @brief Specifies suitable scalar types for EucVec.
 *
 * Ensures that the scalar type is:
 * - [Equality
 * comparable](https://en.cppreference.com/w/cpp/concepts/equality_comparable)
 * - Has definitions for the arithmetic operators `+, -, *, /, +=, -=, *=, /=`
 * - Is closed under the operators `+, -, *, /`
 */
// clang-format off
template <typename ScalarT>
concept AcceptableScalar = std::equality_comparable<ScalarT> && requires(ScalarT a, ScalarT b) {
    // Closure under arithmetic operations
    { a + b } -> std::convertible_to<ScalarT>;
    { a - b } -> std::convertible_to<ScalarT>;
    { a * b } -> std::convertible_to<ScalarT>;
    { a / b } -> std::convertible_to<ScalarT>;
    { -a } -> std::convertible_to<ScalarT>;

    // Check for existence of +=, -=, *=, /=
    a += b; 
    a -= b;
    a *= b;
    a /= b;
};
// clang-format on

/**
 * @brief Specifies real number types for EucVec.
 *
 * In most computer architectures, integer and floating point types
 * represent an adequate subset of real numbers.
 */
template <typename R>
concept RealNumber = std::integral<R> || std::floating_point<R>;

/**
 * @brief Specifies types that support std::abs() in header <cmath>
 *
 * Used in calculating norm of EucVec
 */
template <typename T>
concept AbsComputable = requires(T a) {
    { std::abs(a) } -> std::convertible_to<T>;
};
/// @} concepts

/// @defgroup helper_functions Helper functions
/// @{
template <AbsComputable T>
static constexpr auto abs_val =
    [](const T &scalar) -> T { return std::abs(scalar); };

template <AcceptableScalar ScalarT>
static constexpr auto square =
    [](const ScalarT &scalar) -> ScalarT { return scalar * scalar; };

template <AcceptableScalar ScalarT, std::integral I>
static constexpr auto int_pow = [](ScalarT base, I exponent) -> ScalarT {
    ScalarT result = 1;
    while (exponent > 0) {
        if (exponent % 2 != 0) {
            result *= base;
        }
        base *= base;
        exponent /= 2;
    }
    return result;
};
/// @} helper_functions

/**
 * @brief A Euclidean vector with scalar type S and fixed size N
 *
 * - Supports basic Euclidean vector operations
 * - Type flexibility, using C++20 concepts to check for valid scalar types
 * - Dimension flexibility in favor of performance
 * - Supports compile-time calculations with constexpr
 * - Built on std::array
 * - Heavy use of standard library functions
 *
 * @tparam ScalarT Scalar type which must be equality-comparable and closed
 * under arithmetic operations (see @ref AcceptableScalar)
 * @tparam N Size of the Euclidean vector as std::size_t
 */
template <AcceptableScalar ScalarT, std::size_t N>
class EucVec {
public:
    /// @defgroup member_types Member types
    /// @{
    using container = std::array<ScalarT, N>;
    using value_type = ScalarT;
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
    /// @} member_types

    /// @defgroup constructors Constructors
    /// @{
    constexpr EucVec() noexcept = default;
    constexpr EucVec(const EucVec &) = default;
    constexpr EucVec(EucVec &&) noexcept = default;
    constexpr EucVec(std::initializer_list<ScalarT> init) {
        std::move(init.begin(), init.end(), m_data.begin());
    };
    /// @} constructors

    /// @defgroup destructors Destructors
    /// @{
    constexpr ~EucVec() noexcept = default;
    /// @} destructors

    /// @defgroup assignment Assignment
    /// @{
    constexpr EucVec &operator=(const EucVec &) = default;
    constexpr EucVec &operator=(EucVec &&) noexcept = default;
    /// @} assignment

    /// @defgroup comparison Comparison
    /// @{
    constexpr bool operator==(const EucVec &) const = default;
    constexpr bool operator!=(const EucVec &) const = default;
    /// @} comparison

    /// @defgroup element_access Element Access
    /// @{
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
    constexpr pointer data() noexcept { return m_data.data(); }
    constexpr const_pointer *data() const noexcept { return m_data.data(); }
    /// @} element_access

    /// @defgroup iterators Iterators
    /// @{
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
    /// @} iterators

    /// @defgroup capacity Capacity
    /// @{
    [[nodiscard]] constexpr bool empty() const noexcept {
        return m_data.empty();
    }
    [[nodiscard]] constexpr size_type size() const noexcept {
        return m_data.size();
    }
    [[nodiscard]] constexpr size_type max_size() const noexcept {
        return m_data.max_size();
    }
    /// @} capacity

    /// @defgroup container_operations Container Operations
    /// @{
    constexpr void fill(const ScalarT &value) { m_data.fill(value); }
    constexpr void swap(EucVec &other) noexcept { m_data.swap(other.m_data); }
    /// @} container_operations

    /// @defgroup vector_operations Vector Operations
    /// @{
    /**
     * @brief Returns the dot product of this EucVec with another EucVec
     *
     * For EucVec @f$\vec{u}@f$ and @f$\vec{v}@f$ of size @f$n@f$, the dot
     * product is defined as
     * @f$ \langle \vec{u}_1 \cdot \vec{v}_1, \ldots,
     * \vec{u}_n \cdot \vec{v}_n \rangle @f$
     *
     * @param other The other EucVec
     * @return The dot product of this EucVec with the other EucVec
     */
    [[nodiscard]] constexpr ScalarT dot(const EucVec &other) const {
        // Similar to std::inner_product, except can leverage parallelism
        return std::transform_reduce(this->begin(), this->end(), other.begin(),
                                     ScalarT{});
    }

    /**
     * @brief Returns the cross product of this EucVec with another EucVec
     *
     * Only defined for vectors with size 3. See
     * [wikipedia](https://en.wikipedia.org/wiki/Cross_product) for more
     * information.
     *
     * @param other The other EucVec
     * @return The cross product of this EucVec with another EucVec
     */
    [[nodiscard]] constexpr EucVec cross(const EucVec &other) const
        requires(N == 3)
    {
        return EucVec<ScalarT, 3>(
            {m_data[1] * other[2] - m_data[2] * other[1],
             m_data[2] * other[0] - m_data[0] * other[2],
             m_data[0] * other[1] - m_data[1] * other[0]});
    }

    /**
     * @brief Makes this EucVec the cross product of this and another EucVec
     *
     * Only defined for vectors with size 3. See
     * [wikipedia](https://en.wikipedia.org/wiki/Cross_product) for more
     * information.
     *
     * @warning Modifies this EucVec
     * @param other The other EucVec
     * @return Reference to this EucVec which has become the cross product
     */
    [[nodiscard]] constexpr EucVec &cross_in_place(const EucVec &other) const
        requires(N == 3)
    {
        m_data[0] = m_data[1] * other[2] - m_data[2] * other[1];
        m_data[1] = m_data[2] * other[0] - m_data[0] * other[2];
        m_data[2] = m_data[0] * other[1] - m_data[1] * other[0];

        return *this;
    }
    /// @} vector_operations

    /// @defgroup vector_norms Vector norms
    /// @{
    /**
     * @brief Calculates the p-norm of this EucVec for positive integer p
     *
     * For EucVec @f$\vec{v}@f$ of size @f$n@f$ and real number @f$p@f$ where
     * @f$p > 0 @f$, the @f$p@f$-norm of @f$\vec{v}@f$ is defined as
     * @f$ \sqrt[p]{ |\vec{v}_1|^p + \cdots + |\vec{v}_n|^p } @f$.
     *
     * Note that for any @f$0 \leq p<1@f$, the above definition still makes
     * sense, but the result is no longer a "norm" as it fails the triangle
     * inequality.
     *
     * For more info, see https://ncatlab.org/nlab/show/p-norm#Generalizations.
     *
     * @warning Instantiates a new instance of norm() for each distinct value of
     * p used in your program.
     * @tparam p Integer value that must be greater than 0
     * @return The p-norm of this EucVec
     */
    template <long long p = 2>
    [[nodiscard]] constexpr ScalarT norm() const
        requires RealNumber<ScalarT> && (p > 0)
    {
        // Constexpr branches determined at compile time
        if constexpr (p == 1) {
            return std::sqrt(
                std::transform_reduce(this->begin(), this->end(), ScalarT{},
                                      std::plus<ScalarT>(), std::abs));
        } else if constexpr (p == 2) {
            return std::sqrt(
                std::transform_reduce(this->begin(), this->end(), ScalarT{},
                                      std::plus<ScalarT>(), square<ScalarT>));
        } else if constexpr (p == std::numeric_limits<ScalarT>::max()) {
            return *std::max_element(this->begin(), this->end());
        } else if constexpr (p % 2 == 0) {
            return std::pow(
                std::transform_reduce(
                    this->begin(), this->end(), ScalarT{}, std::plus<ScalarT>(),
                    [](const ScalarT &s) {
                        return int_pow<ScalarT, long long>(s, p);
                    }),
                1.0L / static_cast<long double>(p));
        } else if constexpr (p % 2 != 0) {
            return std::pow(
                std::transform_reduce(
                    this->begin(), this->end(), ScalarT{}, std::plus<ScalarT>(),
                    [](const ScalarT &s) {
                        return std::abs(int_pow<ScalarT, long long>(s, p));
                    }),
                1.0L / static_cast<long double>(p));
        } else {
            __builtin_unreachable();
        }
    }

    /**
     * @brief Returns the length of this EucVec.
     *
     * Identical to EucVec::norm<2>
     *
     * @return The length of this EucVec.
     * @see norm
     */
    [[nodiscard]] constexpr ScalarT length() const { return this->norm<2>(); }

    /**
     * @brief Returns the p-norm of this EucVec
     *
     * For EucVec @f$\vec{v}@f$ of size @f$n@f$ and real number @f$p@f$ where
     * @f$p > 0 @f$, the @f$p@f$-norm of @f$\vec{v}@f$ is defined as
     * @f$ \sqrt[p]{ |\vec{v}_1|^p + \cdots + |\vec{v}_n|^p } @f$
     *
     * Note that for any @f$0 \leq p<1@f$, the above definition still makes
     * sense, but the result is no longer a "norm" as it fails the triangle
     * inequality.
     *
     * @param p Some real number p where p > 0
     * @return The p-norm of this EucVec
     */
    template <RealNumber R>
    [[nodiscard]] constexpr ScalarT pnorm(R p) const
        requires RealNumber<ScalarT>
    {
        static_assert(p > 0);
        return std::pow(
            std::transform_reduce(
                this->begin(), this->end(), ScalarT{}, std::plus<ScalarT>(),
                [p](const ScalarT &val) { return std::pow(std::abs(val), p); }),
            static_cast<ScalarT>(1.0) / p);
    }

    /**
     * @brief Returns the normalized version of this EucVec (using EucVec::norm)
     *
     * Divides this EucVec by its length, normalizing it. Does not modify this
     * EucVec, instead returning the normalized version of this EucVec
     *
     * @return The normalized version of this EucVec
     */
    template <long long p = 2>
    [[nodiscard]] constexpr EucVec normalize() const {
        return *this / this->norm<p>();
    }

    /**
     * @brief Returns the normalized version of this EucVec (using
     * EucVec::pnorm)
     *
     * Divides this EucVec by its length, normalizing it. Does not modify this
     * EucVec, instead returning the normalized version of this EucVec
     *
     * @return The normalized version of this EucVec
     */
    template <RealNumber R>
    [[nodiscard]] constexpr EucVec pnormalize(R p) const {
        return *this / this->pnorm(p);
    }

    /**
     * @brief Normalizes this EucVec (using EucVec::norm)
     *
     * Divides this EucVec by its Euclidean norm, normalizing it
     *
     * @warning Modifies this EucVec
     * @return Reference to this EucVec which has been normalized
     */
    template <long long p = 2>
    constexpr EucVec &normalize_in_place() {
        return *this /= this->norm<p>();
    }

    /**
     * @brief Normalizes this EucVec (using EucVec::pnorm)
     *
     * Divides this EucVec by its Euclidean norm, normalizing it
     *
     * @warning Modifies this EucVec
     * @return Reference to this EucVec which has been normalized
     */
    template <RealNumber R>
    constexpr EucVec &pnormalize_in_place(R p) {
        return *this /= this->pnorm(p);
    }

    /// @}

    /// @defgroup vector_geometry Vector geometry
    /// @{
    /**
     * @brief Returns the distance between this EucVec and another EucVec (using
     * EucVec::norm)
     *
     * Assuming both EucVecs represent a position vector in @f$n@f$-dimensioanl
     * space
     *
     * @param other The other EucVec
     * @return A scalar of the distance between the two EucVecs
     */
    template <long long p = 2>
    [[nodiscard]] constexpr ScalarT dist_to(const EucVec &other) const {
        return (*this - other).template norm<p>();
    }

    /**
     * @brief Returns the distance between this EucVec and another EucVec (using
     * EucVec::pnorm)
     *
     * Assuming both EucVecs represent a position vector in @f$n@f$-dimensioanl
     * space
     *
     * @param other The other EucVec
     * @return A scalar of the distance between the two EucVecs
     */
    template <RealNumber R>
    [[nodiscard]] constexpr ScalarT pdist_to(const EucVec &other, R p) const {
        return (*this - other).pnorm(p);
    }

    /**
     * @brief Returns the angle in radians between this and another EucVec
     *
     * @param other The other EucVec
     * @return The angle in radians between this and the other EucVec
     */
    [[nodiscard]] constexpr ScalarT radians_between(const EucVec &other) const {
        return std::acos(this->dot(other) / (this->norm() * other.norm()));
    }

    /**
     * @brief Returns the angle in degrees between this and another EucVec
     *
     * @param other The other EucVec
     * @return The angle in degrees between this and the other EucVec
     */
    [[nodiscard]] constexpr ScalarT degrees_between(const EucVec &other) const {
        return this->radians_between(other) * 180.0 / std::numbers::pi;
    }

    /**
     * @brief Returns the scalar projection of this EucVec onto another EucVec
     *
     * @param other The other EucVec
     * @return The scalar projection of this EucVec onto the other EucVec
     */
    [[nodiscard]] constexpr ScalarT scalar_project_onto(
        const EucVec &other) const {
        return this->dot(other) / other.norm();
    }

    /**
     * @brief Returns the vector projection of this EucVec onto another EucVec
     *
     * Does not modify this EucVec
     *
     * @param other The other EucVec
     * @return The vector projection of this EucVec onto the other EucVec
     */
    [[nodiscard]] constexpr EucVec project_onto(const EucVec &other) const {
        return (other.dot(*this) / other.dot(other)) * other;
    }

    /**
     * @brief Projects this EucVec onto another EucVec
     * @warning Modifies this EucVec
     * @param other The other EucVec
     * @return Reference to this EucVec which has been projected
     */
    [[nodiscard]] constexpr EucVec &project_onto_in_place(
        const EucVec &other) const {
        return (*this = (other.dot(*this) / other.dot(other)) * other);
    }

    /**
     * @brief Returns whether this EucVec is orthogonal to another EucVec
     *
     * Two vectors @f$\vec{u}@f$ and @f$\vec{v}@f$ are orthogonal if
     * @f$ \vec{u} \cdot \vec{v} = 0 @f$
     *
     * Assumes that the default value for ScalarT represents the additive
     * identity scalar (i.e. 0).
     *
     * @param other The other EucVec
     * @return true if the two EucVecs are orthogonal
     * @return false otherwise
     */
    [[nodiscard]] constexpr bool is_orthogonal_to(const EucVec &other) const {
        return this->dot(other) == ScalarT{};
    }

    /**
     * @brief Returns whether this EucVec is parallel to another EucVec
     *
     * Although the idea of parallelism can generalize to vector spaces over
     * non-numerical fields, this implementation relies on numbers.
     *
     * @param other The other EucVec
     * @return true if the two EucVecs are parallel
     * @return false otherwise
     */
    [[nodiscard]] constexpr bool is_parallel_to(const EucVec &other) const
        requires RealNumber<ScalarT>
    {
        const ScalarT radians = this->radians_between(other);
        return radians == static_cast<ScalarT>(0) ||
               radians == static_cast<ScalarT>(std::numbers::pi);
    }
    /// @}

private:
    container m_data;
};

/// @defgroup arithmetic_operations Arithmetic Operations
/// @{
/**
 * @brief Vector addition
 * @return The sum of the two EucVecs
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator+(const EucVec<ScalarT, N> &lhs,
                                       const EucVec<ScalarT, N> &rhs) {
    EucVec<ScalarT, N> sum;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), sum.begin(),
                   std::plus<ScalarT>());
    return sum;
}

/**
 * @brief Vector addition
 * @return Reference to the left-hand EucVec which has become the sum
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> &operator+=(EucVec<ScalarT, N> &lhs,
                                         const EucVec<ScalarT, N> &rhs) {
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                   std::plus<ScalarT>());
    return lhs;
}

/**
 * @brief Vector subtraction
 * @return The difference of the two EucVecs
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator-(const EucVec<ScalarT, N> &lhs,
                                       const EucVec<ScalarT, N> &rhs) {
    EucVec<ScalarT, N> diff;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), diff.begin(),
                   std::minus<ScalarT>());
    return diff;
}

/**
 * @brief Vector subtraction
 * @return Reference to the left-hand EucVec which has become the diference
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> &operator-=(EucVec<ScalarT, N> &lhs,
                                         const EucVec<ScalarT, N> &rhs) {
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(),
                   std::minus<ScalarT>());
    return lhs;
}

/**
 * @brief Scalar multiplication
 * @return The product of the scalar and EucVec
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator*(const ScalarT &scalar,
                                       const EucVec<ScalarT, N> &vec) {
    EucVec<ScalarT, N> prod;
    std::transform(vec.begin(), vec.end(), prod.begin(),
                   [scalar](const ScalarT &val) { return val * scalar; });
    return prod;
}

/**
 * @brief Scalar multiplication
 * @return The product of the EucVec and scalar
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator*(const EucVec<ScalarT, N> &vec,
                                       const ScalarT &scalar) {
    return scalar * vec;
}

/**
 * @brief Scalar multiplication
 * @return Reference to the left-hand EucVec which has become the product
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> &operator*=(EucVec<ScalarT, N> &vec,
                                         const ScalarT &scalar) {
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [scalar](const ScalarT &val) { return val * scalar; });
    return vec;
}

/**
 * @brief Unary negation
 * @return The product of the EucVec and scalar -1
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator-(const EucVec<ScalarT, N> &vec) {
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [](const ScalarT &val) { return -val; });
    return vec;
}

/**
 * @brief Scalar division
 * @return The quotient of the EucVec and scalar
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator/(const EucVec<ScalarT, N> &vec,
                                       const ScalarT &scalar) {
    EucVec<ScalarT, N> quot;
    std::transform(vec.begin(), vec.end(), quot.begin(),
                   [scalar](const ScalarT &val) { return val / scalar; });
    return quot;
}

/**
 * @brief Scalar division
 * @return Reference to the left-hand EucVec which has become the quotient
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> &operator/=(EucVec<ScalarT, N> &vec,
                                         const ScalarT &scalar) {
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [scalar](const ScalarT &val) { return val / scalar; });
    return vec;
}
/// @}

#endif  // #ifndef EUCLIDEAN_VECTOR_HPP
