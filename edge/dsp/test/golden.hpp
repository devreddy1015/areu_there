// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — the shared golden-vector loader. HOST TEST CODE ONLY.
//
// This header is NOT part of the library and never ships to the node: it uses
// the heap, stdio and std::string freely, none of which are allowed in
// `edge/dsp/*.hpp`.
//
// Vector format (spec/wire.md §3, written by spec/golden/generate.py):
//
//   NAME.kv            one "key value" per line. Scalars, then "array.X N"
//                      declaring a companion array of N elements.
//   NAME__X.f64        array X as raw little-endian float64, N elements.
//   NAME.json          the same data for Rust and tools. Unused here; C++
//                      reads the .kv so the test binary needs no JSON parser.
//
// Values may be "inf", "-inf" or "nan" and are parsed as such
// (presence.kv carries threshold_before_calibration = inf).
//
// ---------------------------------------------------------------------------
// A MISSING VECTOR IS A HARD FAILURE, NEVER A SKIP
// ---------------------------------------------------------------------------
// A parity test that silently passes because it could not find its data is the
// worst possible outcome in this project: it reports green while the port is
// unverified. Every path that cannot produce the requested data prints a
// banner to stderr and exits non-zero. There is deliberately no "vector not
// found, skipping" branch anywhere in this file.
//
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace saans {
namespace golden {

// Default matches the sibling test harness's convention (`edge/dsp/Makefile`
// passes the directory as argv[1]; this is the fallback when nothing else
// says). Not an #error: a test built by either of the two Makefiles in this
// tree must compile, and a wrong-but-present path still fails loudly at load
// time rather than silently skipping.
#ifndef SAANS_GOLDEN_DIR
#define SAANS_GOLDEN_DIR "../../../spec/golden"
#endif

namespace detail {
inline std::string& dir_override() {
    static std::string d;
    return d;
}
}  // namespace detail

/// Where the vectors live. Resolution order, most specific first:
///   1. the path passed on the command line (see `set_dir` / `dir_from_argv`)
///   2. $SAANS_GOLDEN_DIR
///   3. -DSAANS_GOLDEN_DIR from the build
inline std::string dir() {
    if (!detail::dir_override().empty()) {
        return detail::dir_override();
    }
    const char* env = std::getenv("SAANS_GOLDEN_DIR");
    return (env != nullptr && env[0] != '\0') ? std::string(env) : std::string(SAANS_GOLDEN_DIR);
}

inline void set_dir(const std::string& d) { detail::dir_override() = d; }

/// `main` helper: both Makefiles in this tree should drive any test the same
/// way, so every test accepts an optional vector directory as argv[1].
inline void dir_from_argv(int argc, char** argv) {
    if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
        set_dir(argv[1]);
    }
}

/// Loud, non-skippable failure.
[[noreturn]] inline void fatal(const char* fmt, ...) {
    std::fflush(stdout);
    std::fprintf(stderr,
                 "\n"
                 "################################################################\n"
                 "## GOLDEN VECTOR FAILURE - THE TEST DID NOT RUN               ##\n"
                 "################################################################\n"
                 "## ");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr,
                 "\n"
                 "## Vector directory: %s\n"
                 "## Regenerate with: python3 spec/golden/generate.py\n"
                 "## (or point $SAANS_GOLDEN_DIR at the right place)\n"
                 "################################################################\n\n",
                 dir().c_str());
    std::exit(2);
}

inline bool host_is_little_endian() {
    const std::uint32_t one = 1u;
    unsigned char       b[4];
    std::memcpy(b, &one, 4);
    return b[0] == 1u;
}

/// One loaded vector set: the scalars from NAME.kv plus every array it
/// declares, read from the companion NAME__FIELD.f64 files.
class Vectors {
  public:
    explicit Vectors(const std::string& name) : name_(name) { load(); }

    const std::string& name() const { return name_; }

    bool has(const std::string& key) const { return raw_.find(key) != raw_.end(); }

    /// Scalar as a double. Missing key or non-numeric value is fatal — a test
    /// that reads a field the generator stopped emitting must not quietly
    /// compare against zero.
    double scalar(const std::string& key) const {
        const auto it = raw_.find(key);
        if (it == raw_.end()) {
            fatal("%s.kv has no scalar '%s'", name_.c_str(), key.c_str());
        }
        const std::string& v = it->second;
        const char*        s = v.c_str();
        char*              end = nullptr;
        // strtod parses "inf", "-inf", "nan" and "infinity" per C99, which is
        // exactly the set repr() can produce for a non-finite Python float.
        const double d = std::strtod(s, &end);
        if (end == s) {
            fatal("%s.kv: scalar '%s' is not a number: '%s'", name_.c_str(), key.c_str(), s);
        }
        while (*end == ' ' || *end == '\t' || *end == '\r') {
            ++end;
        }
        if (*end != '\0') {
            fatal("%s.kv: trailing text after scalar '%s': '%s'", name_.c_str(), key.c_str(), s);
        }
        return d;
    }

    float  scalar_f32(const std::string& key) const { return static_cast<float>(scalar(key)); }
    bool   flag(const std::string& key) const { return scalar(key) != 0.0; }
    long long integer(const std::string& key) const {
        return static_cast<long long>(scalar(key));
    }

    /// Array as doubles, exactly the bytes the reference wrote. Missing array
    /// is fatal.
    const std::vector<double>& array(const std::string& key) const {
        const auto it = arrays_.find(key);
        if (it == arrays_.end()) {
            fatal("%s.kv declares no array '%s'", name_.c_str(), key.c_str());
        }
        return it->second;
    }

    /// Array narrowed to float32 — the input side of every parity test, since
    /// the node only ever sees float32.
    std::vector<float> array_f32(const std::string& key) const {
        const std::vector<double>& src = array(key);
        std::vector<float>         out(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            out[i] = static_cast<float>(src[i]);
        }
        return out;
    }

  private:
    void load() {
        if (!host_is_little_endian()) {
            fatal("host is big-endian; the .f64 vectors are little-endian");
        }
        const std::string path = dir() + "/" + name_ + ".kv";
        std::FILE*        fh   = std::fopen(path.c_str(), "r");
        if (fh == nullptr) {
            fatal("cannot open %s", path.c_str());
        }

        std::vector<std::pair<std::string, std::size_t>> declared;
        char                                            line[4096];
        int                                             lineno = 0;
        while (std::fgets(line, sizeof(line), fh) != nullptr) {
            ++lineno;
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
                s.pop_back();
            }
            if (s.empty()) {
                continue;
            }
            const std::size_t sp = s.find(' ');
            if (sp == std::string::npos) {
                std::fclose(fh);
                fatal("%s:%d: expected 'key value', got '%s'", path.c_str(), lineno, s.c_str());
            }
            const std::string key = s.substr(0, sp);
            const std::string val = s.substr(sp + 1);

            if (key.rfind("array.", 0) == 0) {
                const std::string    field = key.substr(6);
                const long long      count = std::strtoll(val.c_str(), nullptr, 10);
                if (count < 0) {
                    std::fclose(fh);
                    fatal("%s:%d: negative array length for '%s'", path.c_str(), lineno,
                          field.c_str());
                }
                declared.emplace_back(field, static_cast<std::size_t>(count));
            } else {
                raw_[key] = val;
            }
        }
        std::fclose(fh);

        for (const auto& d : declared) {
            arrays_[d.first] = load_f64(d.first, d.second);
        }
    }

    std::vector<double> load_f64(const std::string& field, std::size_t count) const {
        const std::string path = dir() + "/" + name_ + "__" + field + ".f64";
        std::FILE*        fh   = std::fopen(path.c_str(), "rb");
        if (fh == nullptr) {
            // The .kv promised this array. Its absence means the vector set is
            // half-written or half-copied, and continuing would test nothing.
            fatal("%s.kv declares array '%s' (%zu elements) but %s is missing", name_.c_str(),
                  field.c_str(), count, path.c_str());
        }
        std::fseek(fh, 0, SEEK_END);
        const long bytes = std::ftell(fh);
        std::fseek(fh, 0, SEEK_SET);
        if (bytes < 0 || static_cast<std::size_t>(bytes) != count * sizeof(double)) {
            std::fclose(fh);
            fatal("%s: expected %zu bytes (%zu float64), file holds %ld", path.c_str(),
                  count * sizeof(double), count, bytes);
        }
        std::vector<double> out(count);
        if (count > 0) {
            const std::size_t got = std::fread(out.data(), sizeof(double), count, fh);
            if (got != count) {
                std::fclose(fh);
                fatal("%s: short read, %zu of %zu float64", path.c_str(), got, count);
            }
        }
        std::fclose(fh);
        return out;
    }

    std::string                                  name_;
    std::map<std::string, std::string>           raw_;
    std::map<std::string, std::vector<double>>   arrays_;
};

// ---------------------------------------------------------------------------
// Comparison and reporting
// ---------------------------------------------------------------------------
//
// WHAT "RELATIVE" MEANS FOR A FILTERED SEQUENCE.
//
// SAANS_PLAN §4 asks for 1e-4 relative on filter outputs. A bandpass output
// crosses zero on every cycle, so a strict per-sample |a-b|/|b| is unbounded
// there for reasons that have nothing to do with the port being right: at a
// sample 1e-7 of full scale, float32 cannot represent a 1e-4 relative
// agreement at all. The metric asserted here is therefore
//
//     dev = max_i |got_i - exp_i| / max_i |exp_i|
//
// i.e. the worst error as a fraction of the sequence's own full scale. The
// per-sample relative deviation is ALSO computed and printed, restricted to
// samples above a stated fraction of full scale, so the margin is visible and
// nothing is hidden behind the choice of denominator. Scalars are compared
// with a plain relative deviation; rates in absolute BPM.
//
// No helper here ever widens a tolerance on its own. A failure prints the
// worst offender with its index and both values.

struct ArrayDeviation {
    double      full_scale_rel = 0.0;  ///< max|err| / max|expected|   (ASSERTED)
    double      strong_rel     = 0.0;  ///< max |err|/|exp| over samples >= gate_fraction * FS
    double      pointwise_rel  = 0.0;  ///< max |err|/|exp| over every non-zero sample
    double      rms_rel        = 0.0;  ///< rms(err) / rms(expected)
    double      max_abs        = 0.0;
    double      full_scale     = 0.0;
    std::size_t worst_index    = 0;
    double      worst_got      = 0.0;
    double      worst_expected = 0.0;
    double      gate           = 0.0;  ///< |expected| threshold used for strong_rel
    std::size_t gated_samples  = 0;    ///< how many samples cleared the gate
    std::size_t pointwise_index = 0;   ///< where pointwise_rel was attained

    /// The magnitude below which a sample CANNOT meet `tol` relative, given the
    /// absolute error floor this series carries. Reported as a fraction of full
    /// scale: it is the honest answer to "where does per-sample relative stop
    /// being a meaningful contract for this sequence?".
    double crossover_fraction(double tol) const {
        if (full_scale <= 0.0 || tol <= 0.0) {
            return 0.0;
        }
        return (max_abs / tol) / full_scale;
    }
};

/// Compare a float32 result series against the f64 reference series.
/// `gate_fraction` sets which samples count toward `strong_rel` (default: those
/// at or above half of full scale).
inline ArrayDeviation compare(const float* got, const double* expected, std::size_t n,
                              double gate_fraction = 0.5) {
    ArrayDeviation d;
    if (n == 0) {
        return d;
    }
    for (std::size_t i = 0; i < n; ++i) {
        d.full_scale = std::fmax(d.full_scale, std::fabs(expected[i]));
    }
    d.gate = d.full_scale * gate_fraction;

    double sq_err = 0.0;
    double sq_ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double g   = static_cast<double>(got[i]);
        const double e   = expected[i];
        const double err = std::fabs(g - e);
        sq_err += err * err;
        sq_ref += e * e;
        if (err > d.max_abs) {
            d.max_abs        = err;
            d.worst_index    = i;
            d.worst_got      = g;
            d.worst_expected = e;
        }
        if (std::fabs(e) > 0.0) {
            const double rel = err / std::fabs(e);
            if (rel > d.pointwise_rel) {
                d.pointwise_rel   = rel;
                d.pointwise_index = i;
            }
            if (std::fabs(e) >= d.gate) {
                ++d.gated_samples;
                d.strong_rel = std::fmax(d.strong_rel, rel);
            }
        }
    }
    d.full_scale_rel = (d.full_scale > 0.0) ? d.max_abs / d.full_scale : d.max_abs;
    d.rms_rel        = (sq_ref > 0.0) ? std::sqrt(sq_err / sq_ref) : std::sqrt(sq_err);
    return d;
}

/// Plain relative deviation for a scalar; falls back to absolute when the
/// reference value is zero (nothing else is meaningful there).
inline double rel_dev(double got, double expected) {
    const double err = std::fabs(got - expected);
    return (expected != 0.0) ? err / std::fabs(expected) : err;
}

/// Accumulates one test's checks, prints each line with the margin it had, and
/// returns a process exit code. Every check prints — a silent pass tells you
/// nothing about whether the comparison actually happened.
class Report {
  public:
    void begin(const std::string& title, const std::string& vector_name) {
        std::printf("\n== %s  [%s]\n", title.c_str(), vector_name.c_str());
    }

    void note(const char* fmt, ...) {
        std::printf("   .. ");
        va_list ap;
        va_start(ap, fmt);
        std::vprintf(fmt, ap);
        va_end(ap);
        std::printf("\n");
    }

    /// Relative-tolerance scalar check.
    void close(const std::string& what, double got, double expected, double tol) {
        const double dev = rel_dev(got, expected);
        record(dev <= tol, what,
               "got %.9g  expect %.9g  rel dev %.3g  (tol %.3g)", got, expected, dev, tol);
        if (tol > 0.0) {
            worst_rel_ = std::fmax(worst_rel_, dev / tol);
        }
    }

    /// Absolute-tolerance check — rates in BPM, counters within 1.
    void close_abs(const std::string& what, double got, double expected, double tol) {
        const double dev = std::fabs(got - expected);
        record(dev <= tol, what, "got %.9g  expect %.9g  abs dev %.3g  (tol %.3g)", got,
               expected, dev, tol);
    }

    void equal(const std::string& what, long long got, long long expected) {
        record(got == expected, what, "got %lld  expect %lld  (exact)", got, expected);
    }

    void truth(const std::string& what, bool ok, const std::string& detail) {
        record(ok, what, "%s", detail.c_str());
    }

    /// Series check. Two assertions, both at `tol`:
    ///   1. the worst error as a fraction of the series' own full scale, and
    ///   2. the worst per-sample relative error among samples at or above half
    ///      of full scale — the strictest per-sample claim that is meaningful
    ///      for a sequence that crosses zero.
    /// The unrestricted per-sample figure and the magnitude below which `tol`
    /// becomes unreachable are printed but not asserted, so the shape of the
    /// error is visible rather than hidden behind the choice of denominator.
    void close_array(const std::string& what, const float* got, const double* expected,
                     std::size_t n, double tol) {
        if (n == 0) {
            record(false, what, "empty series - nothing was compared");
            return;
        }
        const ArrayDeviation d  = compare(got, expected, n);
        const bool           ok = (d.full_scale_rel <= tol) && (d.strong_rel <= tol);
        record(ok, what,
               "n=%zu  full-scale rel %.3g | strong-sample rel %.3g over %zu samples >= %.3g "
               "(both tol %.3g)  ||  rms rel %.3g, max abs %.3g of FS %.3g, worst i=%zu "
               "(%.9g vs %.9g); unrestricted per-sample rel %.3g at i=%zu, where tol is "
               "unreachable below %.2f%% of FS",
               n, d.full_scale_rel, d.strong_rel, d.gated_samples, d.gate, tol, d.rms_rel,
               d.max_abs, d.full_scale, d.worst_index, d.worst_got, d.worst_expected,
               d.pointwise_rel, d.pointwise_index, d.crossover_fraction(tol) * 100.0);
        if (tol > 0.0) {
            worst_rel_ = std::fmax(worst_rel_, d.full_scale_rel / tol);
            worst_rel_ = std::fmax(worst_rel_, d.strong_rel / tol);
        }
    }

    int finish(const std::string& suite) {
        std::printf(
            "\n----------------------------------------------------------------\n");
        if (failed_ == 0) {
            std::printf("%s: PASS  %d/%d checks\n", suite.c_str(), passed_, passed_ + failed_);
            std::printf("worst margin used: %.1f%% of the tightest tolerance\n",
                        worst_rel_ * 100.0);
        } else {
            std::printf("%s: FAIL  %d of %d checks failed\n", suite.c_str(), failed_,
                        passed_ + failed_);
            for (const std::string& f : failures_) {
                std::printf("  FAILED: %s\n", f.c_str());
            }
        }
        std::printf("----------------------------------------------------------------\n");
        std::fflush(stdout);
        return (failed_ == 0) ? 0 : 1;
    }

    int failed() const { return failed_; }

  private:
    void record(bool ok, const std::string& what, const char* fmt, ...) {
        char detail[1024];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(detail, sizeof(detail), fmt, ap);
        va_end(ap);
        if (ok) {
            ++passed_;
            std::printf("   ok   %-34s %s\n", what.c_str(), detail);
        } else {
            ++failed_;
            std::printf("   FAIL %-34s %s\n", what.c_str(), detail);
            failures_.push_back(what + "  (" + detail + ")");
        }
    }

    int                      passed_ = 0;
    int                      failed_ = 0;
    double                   worst_rel_ = 0.0;
    std::vector<std::string> failures_;
};

}  // namespace golden
}  // namespace saans
