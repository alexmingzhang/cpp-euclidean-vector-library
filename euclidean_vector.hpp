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
#include <complex>
#include <concepts>
#include <exception>
#include <limits>
#include <numbers>
#include <numeric>

/// @defgroup concepts Concepts
/// @{
/**
 * @brief Specifies suitable scalar types for EucVec.
 *
 * Ensures that the scalar type is [equality
 * comparable](https://en.cppreference.com/w/cpp/concepts/equality_comparable)
 * and closed under the standard arithmetic operations.
 */
// clang-format off
template <typename ScalarT>
concept AcceptableScalar = std::equality_comparable<ScalarT> && requires(ScalarT a, ScalarT b) {
    { +a } -> std::convertible_to<ScalarT>;
    { -a } -> std::convertible_to<ScalarT>;
    { a + b } -> std::convertible_to<ScalarT>;
    { a - b } -> std::convertible_to<ScalarT>;
    { a * b } -> std::convertible_to<ScalarT>;
    { a / b } -> std::convertible_to<ScalarT>;
    { a += b } -> std::convertible_to<ScalarT>;
    { a -= b } -> std::convertible_to<ScalarT>;
    { a *= b } -> std::convertible_to<ScalarT>;
    { a /= b } -> std::convertible_to<ScalarT>;
};
// clang-format on

/**
 * @brief Specifies real number types for EucVec
 *
 * In most computer architectures, integer and floating point types
 * represent an adequate subset of real numbers.
 */
template <typename R>
concept RealNumber = std::integral<R> || std::floating_point<R>;

/**
 * @brief Specifies complex number types for EucVec
 *
 * Expected to use std::complex for complex numbers
 */
template <typename C>
concept ComplexNumber = requires { typename C::value_type; } &&
                        std::is_same_v<C, std::complex<typename C::value_type>>;

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
template <AcceptableScalar ScalarT>
static constexpr auto square =
    [](const ScalarT &scalar) -> ScalarT { return scalar * scalar; };

template <AcceptableScalar ScalarT, std::integral I>
static constexpr auto int_pow = [](ScalarT base, I exponent) -> ScalarT {
    ScalarT result = ScalarT{1};
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
class EucVec : public std::array<ScalarT, N> {
public:
    /// @defgroup vector_operations Vector Operations
    /// @{
    /**
     * @brief Returns the dot product of this EucVec with another EucVec
     *
     * For EucVec @f$\vec{u}@f$ and @f$\vec{v}@f$ of size @f$n@f$, the dot
     * product is defined as
     * @f$ \vec{u}_1 \cdot \vec{v}_1 + \cdots + \vec{u}_n \cdot \vec{v}_n @f$
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
        return EucVec<ScalarT, 3>{
            this->operator[](1) * other[2] - this->operator[](2) * other[1],
            this->operator[](2) * other[0] - this->operator[](0) * other[2],
            this->operator[](0) * other[1] - this->operator[](1) * other[0]};
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
    constexpr EucVec &cross_in_place(const EucVec &other)
        requires(N == 3)
    {
        return (*this) = cross(other);
    }
    /// @} vector_operations

    /// @defgroup vector_norms Vector norms
    /// @{
    /**
     * @brief Calculates the p-norm of this EucVec
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
     * p used in your program. For floating-point ScalarT, may return `inf` for
     * large enough p.
     * @tparam p Integer value that must be greater than 0
     * @return The p-norm of this EucVec
     *
     * @pre p > 0
     *
     * @see EucVec::pnorm
     */
    template <long long p = 2>
    [[nodiscard]] constexpr ScalarT norm() const
        requires RealNumber<ScalarT> && (p > 0)
    {
        if constexpr (p == 1) {
            // "Manhattan norm" or "Taxicab norm"
            return std::transform_reduce(this->begin(), this->end(), ScalarT{},
                                         std::plus<ScalarT>(),
                                         std::abs<ScalarT>);
        } else if constexpr (p == 2) {
            // Euclidean norm
            return std::sqrt(
                std::transform_reduce(this->begin(), this->end(), ScalarT{},
                                      std::plus<ScalarT>(), square<ScalarT>));
        } else if constexpr (p % 2 == 0) {
            // Even integer norm; skip absolute value
            return std::pow(
                std::transform_reduce(
                    this->begin(), this->end(), ScalarT{}, std::plus<ScalarT>(),
                    [](const ScalarT &s) {
                        return int_pow<ScalarT, long long>(s, p);
                    }),
                1.0L / static_cast<long double>(p));
        } else if constexpr (p % 2 != 0) {
            // Odd integer norm
            return std::pow(
                std::transform_reduce(
                    this->begin(), this->end(), ScalarT{}, std::plus<ScalarT>(),
                    [](const ScalarT &s) {
                        return std::abs(int_pow<ScalarT, long long>(s, p));
                    }),
                1.0L / static_cast<long double>(p));
        } else {
            throw std::runtime_error("Could not determine norm");
        }
    }

    /**
     * @brief Returns the length of this EucVec.
     *
     * Identical to EucVec::norm<2>
     *
     * @return The length of this EucVec.
     * @see EucVec::norm
     */
    [[nodiscard]] constexpr auto length() const { return this->norm<2>(); }

    /**
     * @brief Returns the infinity norm or supremum norm of this EucVec
     *
     * The infinity norm is defined simply as the supremum of the elements in
     * the vector. In our case with finitely-sized EucVecs, we can simply take
     * the maximum element as the supremum.
     *
     * Identical to EucVec::norm<std::numeric_limits<long long>::max()>
     *
     * @return The maximum element of this EucVec
     */
    [[nodiscard]] constexpr ScalarT infnorm() const {
        return *std::max_element(this->begin(), this->end());
    }

    /**
     * @brief Calculates the p-norm of a complex-valued EucVec
     *
     * @warning Instantiates a new instance of norm() for each distinct value of
     * p used in your program.
     * @tparam p Integer value that must be greater than 0
     * @return The p-norm of this EucVec
     */
    template <long long p = 2>
    [[nodiscard]] constexpr auto norm() const
        requires ComplexNumber<ScalarT> && (p > 0)
    {
        EucVec<typename ScalarT::value_type, N> abs_vec;
        std::transform(this->begin(), this->end(), abs_vec.begin(),
                       std::abs<typename ScalarT::value_type>);
        return abs_vec.template norm<p>();
    }

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
     *
     * @pre p > 0
     * @see EucVec::norm
     */
    template <RealNumber R>
    [[nodiscard]] constexpr auto pnorm(R p) const
        requires RealNumber<ScalarT>
    {
        assert(p > R{0});
        return std::pow(
            std::transform_reduce(
                this->begin(), this->end(), ScalarT{}, std::plus<ScalarT>(),
                [p](const ScalarT &val) { return std::pow(std::abs(val), p); }),
            R{1} / p);
    }

    template <RealNumber R>
    [[nodiscard]] constexpr auto pnorm(R p) const
        requires ComplexNumber<ScalarT>
    {
        assert(p > R{0});
        EucVec<typename ScalarT::value_type, N> abs_vec;
        std::transform(this->begin(), this->end(), abs_vec.begin(),
                       std::abs<typename ScalarT::value_type>);
        return abs_vec.pnorm(p);
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
    constexpr EucVec &project_onto_in_place(const EucVec &other) const {
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
        return radians == ScalarT{} ||
               radians == static_cast<ScalarT>(std::numbers::pi);
    }
    /// @}
};

/// @defgroup arithmetic_operations Arithmetic Operations
/// @{
/**
 * @brief Unary plus operator for EucVec
 * @return A new EucVec with the unary plus operator applied to each of its
 * elements
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator+(const EucVec<ScalarT, N> &vec) {
    EucVec<ScalarT, N> copy;
    std::transform(vec.begin(), vec.end(), copy.begin(),
                   [](const auto &val) { return +val; });
    return copy;
}

/**
 * @brief Unary minus operator for EucVec
 * @return A new EucVec with the unary minus operator applied to each of its
 * elements
 */
template <AcceptableScalar ScalarT, std::size_t N>
constexpr EucVec<ScalarT, N> operator-(const EucVec<ScalarT, N> &vec) {
    EucVec<ScalarT, N> neg;
    std::transform(vec.begin(), vec.end(), neg.begin(),
                   [](const auto &val) { return -val; });
    return neg;
}

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
template <AcceptableScalar ScalarT, std::size_t N,
          AcceptableScalar OtherScalarT>
constexpr EucVec<ScalarT, N> operator*(const OtherScalarT &scalar,
                                       const EucVec<ScalarT, N> &vec) {
    EucVec<ScalarT, N> prod;
    std::transform(vec.begin(), vec.end(), prod.begin(),
                   [scalar](const auto &val) { return val * scalar; });
    return prod;
}

/**
 * @brief Scalar multiplication
 * @return The product of the EucVec and scalar
 */
template <AcceptableScalar ScalarT, std::size_t N,
          AcceptableScalar OtherScalarT>
constexpr EucVec<ScalarT, N> operator*(const EucVec<ScalarT, N> &vec,
                                       const OtherScalarT &scalar) {
    return scalar * vec;
}

/**
 * @brief Scalar multiplication
 * @return Reference to the left-hand EucVec which has become the product
 */
template <AcceptableScalar ScalarT, std::size_t N,
          AcceptableScalar OtherScalarT>
constexpr EucVec<ScalarT, N> &operator*=(EucVec<ScalarT, N> &vec,
                                         const OtherScalarT &scalar) {
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [scalar](const auto &val) { return val * scalar; });
    return vec;
}

/**
 * @brief Scalar division
 * @return The quotient of the EucVec and scalar
 */
template <AcceptableScalar ScalarT, std::size_t N,
          AcceptableScalar OtherScalarT>
constexpr EucVec<ScalarT, N> operator/(const EucVec<ScalarT, N> &vec,
                                       const OtherScalarT &scalar) {
    EucVec<ScalarT, N> quot;
    std::transform(vec.begin(), vec.end(), quot.begin(),
                   [scalar](const auto &val) { return val / scalar; });
    return quot;
}

/**
 * @brief Scalar division
 * @return Reference to the left-hand EucVec which has become the quotient
 */
template <AcceptableScalar ScalarT, std::size_t N,
          AcceptableScalar OtherScalarT>
constexpr EucVec<ScalarT, N> &operator/=(EucVec<ScalarT, N> &vec,
                                         const OtherScalarT &scalar) {
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [scalar](const auto &val) { return val / scalar; });
    return vec;
}
/// @}

#endif  // #ifndef EUCLIDEAN_VECTOR_HPP
