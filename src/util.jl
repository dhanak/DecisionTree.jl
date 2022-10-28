# written by Poom Chiarawongse <eight1911@gmail.com>

module util

    export gini, entropy, zero_one, q_bi_sort!, hypergeometric, check_input

    function assign(Y :: AbstractVector{T}, list :: AbstractVector{T}) where T
        dict = Dict{T, Int}()
        @simd for i in 1:length(list)
            @inbounds dict[list[i]] = i
        end

        _Y = Array{Int}(undef, length(Y))
        @simd for i in 1:length(Y)
            @inbounds _Y[i] = dict[Y[i]]
        end

        return list, _Y
    end

    function assign(Y :: AbstractVector{T}) where T
        set = Set{T}()
        for y in Y
            push!(set, y)
        end
        list = collect(set)
        return assign(Y, list)
    end

    @inline function zero_one(ns, n)
        return 1.0 - maximum(ns) / n
    end

    @inline function gini(ns, n)
        s = 0.0
        @simd for k in ns
            s += k * (n - k)
        end
        return s / (n * n)
    end

    # compute table of values i*log(i) for integers in 0 <= i <= maxvalue
    # where tables[i+1] = i * log(i)
    # (0*log(0) is set to 0 for convenience when computing entropy)
    function compute_entropy_terms(maxvalue)
        entropy_terms = zeros(Float64, maxvalue+1)
        for i in 1:maxvalue
            entropy_terms[i+1] = i * log(i)
        end
        return entropy_terms
    end

    # returns the entropy of ns/n, ns is an array of integers
    # and entropy_terms are precomputed entropy terms
    @inline function entropy(ns::AbstractVector{U}, n, entropy_terms) where {U <: Integer}
        s = 0.0
        for k in ns
            s += entropy_terms[k+1]
        end
        return log(n) - s / n
    end

    @inline function entropy(ns, n)
        s = 0.0
        @simd for k in ns
            if k > 0
                s += k * log(k)
            end
        end
        return log(n) - s / n
    end

    # adapted from the Julia Base.Sort Library
    @inline function partition!(v, w, pivot, region)
        i, j = 1, length(region)
        r_start = region.start - 1
        @inbounds while true
            while w[i] <= pivot; i += 1; end;
            while w[j]  > pivot; j -= 1; end;
            i >= j && break
            ri = r_start + i
            rj = r_start + j
            v[ri], v[rj] = v[rj], v[ri]
            w[i], w[j] = w[j], w[i]
            i += 1; j -= 1
        end
        return j
    end

    # adapted from the Julia Base.Sort Library
    function insert_sort!(v, w, lo, hi, offset)
        @inbounds for i = lo+1:hi
            j = i
            x = v[i]
            y = w[offset+i]
            while j > lo
                if x < v[j-1]
                    v[j] = v[j-1]
                    w[offset+j] = w[offset+j-1]
                    j -= 1
                    continue
                end
                break
            end
            v[j] = x
            w[offset+j] = y
        end
        return v
    end

    @inline function _selectpivot!(v, w, lo, hi, offset)
        @inbounds begin
            mi = (lo+hi)>>>1

            # sort the values in v[lo], v[mi], v[hi]

            if v[mi] < v[lo]
                v[mi], v[lo] = v[lo], v[mi]
                w[offset+mi], w[offset+lo] = w[offset+lo], w[offset+mi]
            end
            if v[hi] < v[mi]
                if v[hi] < v[lo]
                    v[lo], v[mi], v[hi] = v[hi], v[lo], v[mi]
                    w[offset+lo], w[offset+mi], w[offset+hi] = w[offset+hi], w[offset+lo], w[offset+mi]
                else
                    v[hi], v[mi] = v[mi], v[hi]
                    w[offset+hi], w[offset+mi] = w[offset+mi], w[offset+hi]
                end
            end

            # move v[mi] to v[lo] and use it as the pivot
            v[lo], v[mi] = v[mi], v[lo]
            w[offset+lo], w[offset+mi] = w[offset+mi], w[offset+lo]
            v_piv = v[lo]
            w_piv = w[offset+lo]
        end

        # return the pivot
        return v_piv, w_piv
    end

    # adapted from the Julia Base.Sort Library
    @inline function _bi_partition!(v, w, lo, hi, offset)
        pivot, w_piv = _selectpivot!(v, w, lo, hi, offset)
        # pivot == v[lo], v[hi] > pivot
        i, j = lo, hi
        @inbounds while true
            i += 1; j -= 1
            while v[i] < pivot; i += 1; end;
            while pivot < v[j]; j -= 1; end;
            i >= j && break
            v[i], v[j] = v[j], v[i]
            w[offset+i], w[offset+j] = w[offset+j], w[offset+i]
        end
        v[j], v[lo] = pivot, v[j]
        w[offset+j], w[offset+lo] = w_piv, w[offset+j]

        # v[j] == pivot
        # v[k] >= pivot for k > j
        # v[i] <= pivot for i < j
        return j
    end


    # adapted from the Julia Base.Sort Library
    # adapted from the Julia Base.Sort Library
    # this sorts v[lo:hi] and w[offset+lo, offset+hi]
    # simultaneously by the values in v[lo:hi]
    const SMALL_THRESHOLD  = 20
    function q_bi_sort!(v, w, lo, hi, offset)
        @inbounds while lo < hi
            hi-lo <= SMALL_THRESHOLD && return insert_sort!(v, w, lo, hi, offset)
            j = _bi_partition!(v, w, lo, hi, offset)
            if j-lo < hi-j
                # recurse on the smaller chunk
                # this is necessary to preserve O(log(n))
                # stack space in the worst case (rather than O(n))
                lo < (j-1) && q_bi_sort!(v, w, lo, j-1, offset)
                lo = j+1
            else
                j+1 < hi && q_bi_sort!(v, w, j+1, hi, offset)
                hi = j-1
            end
        end
        return v
    end

    # The code function below is a small port from numpy's library
    # library which is distributed under the 3-Clause BSD license.
    # The rest of DecisionTree.jl is released under the MIT license.

    # ported by Poom Chiarawongse <eight1911@gmail.com>

    # this is the code for efficient generation
    # of hypergeometric random variables ported from numpy.random

    const logfact = @. Float64(log(factorial(big(0:125))))
    const halfln2pi = log(2π) / 2
    const D1 = 2 * sqrt(2 / MathConstants.e)
    const D2 = 3 - 2 * sqrt(3 / MathConstants.e)

    function hypergeometric(good, bad, sample, rng)

        @inline function logfactorial(k)
            # Use the lookup table.
            k < length(logfact) && return logfact[k + 1]
            # Use the Stirling series, truncated at the 1/k**3 term.
            # (In a Python implementation of this approximation, the result
            # was within 2 ULP of the best 64 bit floating point value for
            # k up to 10000000.)
            return (k + 0.5)*log(k) - k +
                (halfln2pi + (1.0/k)*(1/12.0 - 1/(360.0*k*k)))
        end

        #  Generate a sample from the hypergeometric distribution.
        #
        #  Assume sample is not greater than half the total.  See below
        #  for how the opposite case is handled.
        #
        #  We initialize the following:
        #      computed_sample = sample
        #      remaining_good = good
        #      remaining_total = good + bad
        #
        #  In the loop:
        #  * computed_sample counts down to 0;
        #  * remaining_good is the number of good choices not selected yet;
        #  * remaining_total is the total number of choices not selected yet.
        #
        #  In the loop, we select items by choosing a random integer in
        #  the interval [0, remaining_total), and if the value is less
        #  than remaining_good, it means we have selected a good one,
        #  so remaining_good is decremented.  Then, regardless of that
        #  result, computed_sample is decremented.  The loop continues
        #  until either computed_sample is 0, remaining_good is 0, or
        #  remaining_total == remaining_good.  In the latter case, it
        #  means there are only good choices left, so we can stop the
        #  loop early and select what is left of computed_sample from
        #  the good choices (i.e. decrease remaining_good by computed_sample).
        #
        #  When the loop exits, the actual number of good choices is
        #  good - remaining_good.
        #
        #  If sample is more than half the total, then initially we set
        #      computed_sample = total - sample
        #  and at the end we return remaining_good (i.e. the loop in effect
        #  selects the complement of the result).
        #
        #  It is assumed that when this function is called:
        #    * good, bad and sample are nonnegative;
        #    * the sum good+bad will not result in overflow;
        #    * sample <= good+bad.
        @inline function hypergeometric_sample(good, bad, sample)
            total = good + bad
            computed_sample = min(sample, total - sample)
            remaining_total = total
            remaining_good = good

            while computed_sample > 0 &&
                remaining_good > 0 &&
                remaining_total > remaining_good
                if rand(rng, 0:remaining_total - 1) < remaining_good
                    # Selected a "good" one, so decrement remaining_good.
                    remaining_good -= 1
                end
                computed_sample -= 1
            end

         if remaining_total == remaining_good
             # Only "good" choices are left.
             remaining_good -= computed_sample
         end

            return if sample > total/2
                remaining_good
            else
                good - remaining_good
            end
        end

        # Generate variates from the hypergeometric distribution
        # using the ratio-of-uniforms method.
        #
        # In the code, the variable names a, b, c, g, h, m, p, q, K, T,
        # U and X match the names used in "Algorithm HRUA" beginning on
        # page 82 of Stadlober's 1989 thesis.
        #
        # It is assumed that when this function is called:
        #   * good, bad and sample are nonnegative;
        #   * the sum good+bad will not result in overflow;
        #   * sample <= good+bad.
        #
        # References:
        # -  Ernst Stadlober's thesis "Sampling from Poisson, Binomial and
        #    Hypergeometric Distributions: Ratio of Uniforms as a Simple and
        #    Fast Alternative" (1989)
        # -  Ernst Stadlober, "The ratio of uniforms approach for generating
        #    discrete random variates", Journal of Computational and Applied
        #    Mathematics, 31, pp. 181-189 (1990).
        @inline function hypergeometric_hrua(good, bad, sample)
            popsize = good + bad
            computed_sample = min(sample, popsize - sample)
            mingoodbad = min(good, bad)
            maxgoodbad = max(good, bad)

            # Variables that do not match Stadlober (1989)
            #   Here               Stadlober
            #   ----------------   ---------
            #   mingoodbad            M
            #   popsize               N
            #   computed_sample       n

            p = mingoodbad / popsize
            q = maxgoodbad / popsize

            #  mu is the mean of the distribution.
            mu = computed_sample * p
            a = mu + 0.5

            # var is the variance of the distribution.
            var = (popsize - computed_sample) *
                computed_sample * p * q / (popsize - 1)
            c = sqrt(var + 0.5)

            # h is 2*s_hat (See Stadlober's thesis (1989), Eq. (5.17); or
            # Stadlober (1990), Eq. 8).  s_hat is the scale of the "table mountain"
            # function that dominates the scaled hypergeometric PMF ("scaled" means
            # normalized to have a maximum value of 1).

            h = D1*c + D2
            m = floor(Int64, (computed_sample + 1) * (mingoodbad + 1) /
                (popsize + 2))
            g = logfactorial(m) +
                logfactorial(mingoodbad - m) +
                logfactorial(computed_sample - m) +
                logfactorial(maxgoodbad - computed_sample + m)

            # b is the upper bound for random samples:
            # ... min(computed_sample, mingoodbad) + 1 is the length of the support.
            # ... floor(a + 16*c) is 16 standard deviations beyond the mean.
            #
            # The idea behind the second upper bound is that values that far out in
            # the tail have negligible probabilities.
            #
            # There is a comment in a previous version of this algorithm that says
            #     "16 for 16-decimal-digit precision in D1 and D2",
            # but there is no documented justification for this value.  A lower value
            # might work just as well, but I've kept the value 16 here.

            b = min(min(computed_sample, mingoodbad) + 1, floor(a + 16*c))

            while true
                U = rand(rng)
                V = rand(rng) # "U star" in Stadlober (1989)
                X = a + h*(V - 0.5) / U

                # fast rejection:
                if X < 0.0 || X >= b
                    continue
                end

                K = floor(Int64, X)
                gp = logfactorial(K) +
                    logfactorial(mingoodbad - K) +
                    logfactorial(computed_sample - K) +
                    logfactorial(maxgoodbad - computed_sample + K)
                T = g - gp

                # fast acceptance:
                if (U*(4.0 - U) - 3.0) <= T
                    break
                end

                # fast rejection:
                if U*(U - T) >= 1
                    continue
                end

                if 2.0*log(U) <= T
                    # acceptance
                    break
                end
            end

            if good > bad
                K = computed_sample - K
            end

            if computed_sample < sample
                K = good - K
            end

            return K
        end

        # Draw a sample from the hypergeometric distribution.
        #
        # It is assumed that when this function is called:
        #   * good, bad and sample are nonnegative;
        #   * the sum good+bad will not result in overflow;
        #   * sample <= good+bad.
        return if sample >= 10 && sample <= good + bad - 10
            # This will use the ratio-of-uniforms method.
            hypergeometric_hrua(good, bad, sample)
        else
            # The simpler implementation is faster for small samples.
            hypergeometric_sample(good, bad, sample)
        end
    end

    function check_input(
            X                   :: AbstractMatrix{S},
            Y                   :: AbstractVector{T},
            W                   :: AbstractVector{U},
            max_features        :: Int,
            max_depth           :: Int,
            min_samples_leaf    :: Int,
            min_samples_split   :: Int,
            min_purity_increase :: Float64) where {S, T, U}
        n_samples, n_features = size(X)
        if length(Y) != n_samples
            throw("dimension mismatch between X and Y ($(size(X)) vs $(size(Y))")
        elseif length(W) != n_samples
            throw("dimension mismatch between X and W ($(size(X)) vs $(size(W))")
        elseif max_depth < -1
            throw("unexpected value for max_depth: $(max_depth) (expected:"
                * " max_depth >= 0, or max_depth = -1 for infinite depth)")
        elseif n_features < max_features
            throw("number of features $(n_features) is less than the number "
                * "of max features $(max_features)")
        elseif max_features < 0
            throw("number of features $(max_features) must be >= zero ")
        elseif min_samples_leaf < 1
            throw("min_samples_leaf must be a positive integer "
                * "(given $(min_samples_leaf))")
        elseif min_samples_split < 2
            throw("min_samples_split must be at least 2 "
                * "(given $(min_samples_split))")
        end
    end

end
