#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "int32_map.h"

namespace py = pybind11;

using i64 = std::int64_t;
using i32 = std::int32_t;

enum class TransitionMapKind : i32 {
    kLazy = 0,
    kUnordered = 1,
    kInt32 = 2,
    kLazyInt32 = 3,
};

i32 checked_i32(i64 value, const char* name) {
    if (value < std::numeric_limits<i32>::min() || value > std::numeric_limits<i32>::max()) {
        throw std::overflow_error(std::string(name) + " does not fit in int32_t");
    }
    return static_cast<i32>(value);
}

TransitionMapKind transition_map_kind_from_name(const std::string& name, bool use_small_dict) {
    if (name.empty()) {
        return use_small_dict ? TransitionMapKind::kLazy : TransitionMapKind::kUnordered;
    }
    if (name == "lazy" || name == "small" || name == "small_dict") {
        return TransitionMapKind::kLazy;
    }
    if (name == "unordered" || name == "normal" || name == "normal_dict") {
        return TransitionMapKind::kUnordered;
    }
    if (name == "int32" || name == "suffix" || name == "suffix_map") {
        return TransitionMapKind::kInt32;
    }
    if (name == "lazy_int32" || name == "lazy_suffix" || name == "lazy_suffix_map") {
        return TransitionMapKind::kLazyInt32;
    }
    throw std::invalid_argument("Unknown transition map type: " + name);
}

std::string transition_map_kind_name(TransitionMapKind kind) {
    switch (kind) {
        case TransitionMapKind::kLazy:
            return "lazy";
        case TransitionMapKind::kUnordered:
            return "unordered";
        case TransitionMapKind::kInt32:
            return "int32";
        case TransitionMapKind::kLazyInt32:
            return "lazy_int32";
    }
    return "unknown";
}

class TransitionMap {
public:
    virtual ~TransitionMap() = default;
    virtual bool contains(i64 key) const = 0;
    virtual i64 get(i64 key) const = 0;
    virtual void set(i64 key, i64 value) = 0;
    virtual std::size_t size() const = 0;
    virtual std::vector<std::pair<i64, i64>> items() const = 0;
    virtual std::size_t memory_usage() const = 0;
    virtual std::unique_ptr<TransitionMap> clone() const = 0;
};

class UnorderedTransitionMap final : public TransitionMap {
public:
    bool contains(i64 key) const {
        i32 k = checked_i32(key, "transition key");
        return map_.find(k) != map_.end();
    }

    i64 get(i64 key) const {
        i32 k = checked_i32(key, "transition key");
        auto it = map_.find(k);
        if (it == map_.end()) {
            return -1;
        }
        return it->second;
    }

    void set(i64 key, i64 value) {
        map_[checked_i32(key, "transition key")] = checked_i32(value, "transition state");
    }

    std::size_t size() const {
        return map_.size();
    }

    std::vector<std::pair<i64, i64>> items() const {
        std::vector<std::pair<i64, i64>> out;
        out.reserve(map_.size());
        for (const auto& item : map_) {
            out.emplace_back(item.first, item.second);
        }
        return out;
    }

    std::size_t memory_usage() const {
        using NodeValue = std::pair<const i32, i32>;
        return sizeof(*this) + map_.bucket_count() * sizeof(void*) +
               map_.size() * (sizeof(NodeValue) + sizeof(void*));
    }

    std::unique_ptr<TransitionMap> clone() const {
        return std::make_unique<UnorderedTransitionMap>(*this);
    }

private:
    std::unordered_map<i32, i32> map_;
};

class Int32TransitionMap final : public TransitionMap {
public:
    bool contains(i64 key) const {
        i32 k = checked_i32(key, "transition key");
        return map_.find(k) != map_.end();
    }

    i64 get(i64 key) const {
        i32 k = checked_i32(key, "transition key");
        auto it = map_.find(k);
        if (it == map_.end()) {
            return -1;
        }
        return it->second;
    }

    void set(i64 key, i64 value) {
        map_[checked_i32(key, "transition key")] = checked_i32(value, "transition state");
    }

    std::size_t size() const {
        return map_.size();
    }

    std::vector<std::pair<i64, i64>> items() const {
        std::vector<std::pair<i64, i64>> out;
        out.reserve(map_.size());
        for (const auto& item : map_) {
            out.emplace_back(item.first, item.second);
        }
        return out;
    }

    std::size_t memory_usage() const {
        return sizeof(*this) + map_.memory_usage() - sizeof(map_);
    }

    std::unique_ptr<TransitionMap> clone() const {
        auto copy = std::make_unique<Int32TransitionMap>();
        for (const auto& item : map_) {
            copy->map_[item.first] = item.second;
        }
        return copy;
    }

private:
    Int32Map<i32> map_;
};

template <std::size_t InlineCapacity>
class LazyTransitionMap final : public TransitionMap {
public:
    LazyTransitionMap() {
        keys_.fill(kEmptyKey);
    }

    bool contains(i64 key) const {
        return get(key) != -1;
    }

    i64 get(i64 key) const {
        i32 k = checked_i32(key, "transition key");
        if (map_) {
            auto it = map_->find(k);
            if (it == map_->end()) {
                return -1;
            }
            return it->second;
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] == k) {
                return values_[i];
            }
        }
        return -1;
    }

    void set(i64 key, i64 value) {
        i32 k = checked_i32(key, "transition key");
        i32 v = checked_i32(value, "transition state");
        if (map_) {
            (*map_)[k] = v;
            return;
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] == k) {
                values_[i] = v;
                return;
            }
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] == kEmptyKey) {
                keys_[i] = k;
                values_[i] = v;
                return;
            }
        }
        map_ = std::make_unique<std::unordered_map<i32, i32>>();
        map_->reserve(InlineCapacity + 1);
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] != kEmptyKey) {
                (*map_)[keys_[i]] = values_[i];
            }
        }
        (*map_)[k] = v;
    }

    std::size_t size() const {
        if (map_) {
            return map_->size();
        }
        std::size_t count = 0;
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            count += keys_[i] != kEmptyKey ? 1 : 0;
        }
        return count;
    }

    std::vector<std::pair<i64, i64>> items() const {
        std::vector<std::pair<i64, i64>> out;
        out.reserve(size());
        if (map_) {
            for (const auto& item : *map_) {
                out.emplace_back(item.first, item.second);
            }
            return out;
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] != kEmptyKey) {
                out.emplace_back(keys_[i], values_[i]);
            }
        }
        return out;
    }

    std::size_t memory_usage() const {
        using NodeValue = std::pair<const i32, i32>;
        std::size_t bytes = sizeof(*this);
        if (map_) {
            bytes += sizeof(*map_) + map_->bucket_count() * sizeof(void*) +
                     map_->size() * (sizeof(NodeValue) + sizeof(void*));
        }
        return bytes;
    }

    std::unique_ptr<TransitionMap> clone() const {
        auto copy = std::make_unique<LazyTransitionMap<InlineCapacity>>();
        copy->keys_ = keys_;
        copy->values_ = values_;
        if (map_) {
            copy->map_ = std::make_unique<std::unordered_map<i32, i32>>(*map_);
        }
        return copy;
    }

private:
    static constexpr i32 kEmptyKey = -1;
    std::array<i32, InlineCapacity> keys_{};
    std::array<i32, InlineCapacity> values_{};
    std::unique_ptr<std::unordered_map<i32, i32>> map_;
};

template <std::size_t InlineCapacity>
class LazyInt32TransitionMap final : public TransitionMap {
public:
    LazyInt32TransitionMap() {
        keys_.fill(kEmptyKey);
    }

    bool contains(i64 key) const {
        return get(key) != -1;
    }

    i64 get(i64 key) const {
        i32 k = checked_i32(key, "transition key");
        if (map_) {
            auto it = map_->find(k);
            if (it == map_->end()) {
                return -1;
            }
            return it->second;
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] == k) {
                return values_[i];
            }
        }
        return -1;
    }

    void set(i64 key, i64 value) {
        i32 k = checked_i32(key, "transition key");
        i32 v = checked_i32(value, "transition state");
        if (map_) {
            (*map_)[k] = v;
            return;
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] == k) {
                values_[i] = v;
                return;
            }
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] == kEmptyKey) {
                keys_[i] = k;
                values_[i] = v;
                return;
            }
        }
        map_ = std::make_unique<Int32Map<i32>>();
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] != kEmptyKey) {
                (*map_)[keys_[i]] = values_[i];
            }
        }
        (*map_)[k] = v;
    }

    std::size_t size() const {
        if (map_) {
            return map_->size();
        }
        std::size_t count = 0;
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            count += keys_[i] != kEmptyKey ? 1 : 0;
        }
        return count;
    }

    std::vector<std::pair<i64, i64>> items() const {
        std::vector<std::pair<i64, i64>> out;
        out.reserve(size());
        if (map_) {
            for (const auto& item : *map_) {
                out.emplace_back(item.first, item.second);
            }
            return out;
        }
        for (std::size_t i = 0; i < InlineCapacity; ++i) {
            if (keys_[i] != kEmptyKey) {
                out.emplace_back(keys_[i], values_[i]);
            }
        }
        return out;
    }

    std::size_t memory_usage() const {
        std::size_t bytes = sizeof(*this);
        if (map_) {
            bytes += map_->memory_usage();
        }
        return bytes;
    }

    std::unique_ptr<TransitionMap> clone() const {
        auto copy = std::make_unique<LazyInt32TransitionMap<InlineCapacity>>();
        copy->keys_ = keys_;
        copy->values_ = values_;
        if (map_) {
            copy->map_ = std::make_unique<Int32Map<i32>>();
            for (const auto& item : *map_) {
                (*copy->map_)[item.first] = item.second;
            }
        }
        return copy;
    }

private:
    static constexpr i32 kEmptyKey = -1;
    std::array<i32, InlineCapacity> keys_{};
    std::array<i32, InlineCapacity> values_{};
    std::unique_ptr<Int32Map<i32>> map_;
};

std::unique_ptr<TransitionMap> make_transition_map(TransitionMapKind kind, i64 lazy_threshold) {
    switch (kind) {
        case TransitionMapKind::kLazy:
            switch (lazy_threshold) {
                case 1:
                    return std::make_unique<LazyTransitionMap<1>>();
                case 2:
                    return std::make_unique<LazyTransitionMap<2>>();
                case 3:
                    return std::make_unique<LazyTransitionMap<3>>();
                case 4:
                    return std::make_unique<LazyTransitionMap<4>>();
                case 5:
                    return std::make_unique<LazyTransitionMap<5>>();
                default:
                    throw std::invalid_argument("lazy_threshold must be in [1, 5]");
            }
        case TransitionMapKind::kUnordered:
            return std::make_unique<UnorderedTransitionMap>();
        case TransitionMapKind::kInt32:
            return std::make_unique<Int32TransitionMap>();
        case TransitionMapKind::kLazyInt32:
            switch (lazy_threshold) {
                case 1:
                    return std::make_unique<LazyInt32TransitionMap<1>>();
                case 2:
                    return std::make_unique<LazyInt32TransitionMap<2>>();
                case 3:
                    return std::make_unique<LazyInt32TransitionMap<3>>();
                case 4:
                    return std::make_unique<LazyInt32TransitionMap<4>>();
                case 5:
                    return std::make_unique<LazyInt32TransitionMap<5>>();
                default:
                    throw std::invalid_argument("lazy_threshold must be in [1, 5]");
            }
    }
    throw std::invalid_argument("Unknown transition map kind");
}

struct State {
    explicit State(
        TransitionMapKind map_kind = TransitionMapKind::kLazy,
        i64 lazy_threshold = 1,
        i64 link_ = -1,
        i64 length_ = 0)
        : next(make_transition_map(map_kind, lazy_threshold)), link(link_), length(length_) {}

    State(const State& other)
        : next(other.next->clone()),
          link(other.link),
          length(other.length),
          cnt_endpos(other.cnt_endpos),
          min_endpos(other.min_endpos) {}

    State& operator=(const State& other) {
        if (this == &other) {
            return *this;
        }
        next = other.next->clone();
        link = other.link;
        length = other.length;
        cnt_endpos = other.cnt_endpos;
        min_endpos = other.min_endpos;
        return *this;
    }

    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    std::unique_ptr<TransitionMap> next;
    i64 link = -1;
    i64 length = 0;
    i64 cnt_endpos = 0;
    i64 min_endpos = 0;
};

struct DynDraftResult {
    std::vector<i64> tokens;
    std::vector<i64> ancestors;
};

struct EdgeStats {
    i64 single_next_states = 0;
    i64 total_states = 0;
    double ratio = 0.0;
};

namespace proto {

constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kWireVarint = 0;
constexpr std::uint32_t kWireFixed64 = 1;
constexpr std::uint32_t kWireLengthDelimited = 2;

void write_varint(std::string& out, std::uint64_t value) {
    while (value >= 0x80) {
        out.push_back(static_cast<char>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

std::uint64_t read_varint(const std::string& data, std::size_t& pos) {
    std::uint64_t value = 0;
    int shift = 0;
    while (pos < data.size() && shift <= 63) {
        std::uint8_t byte = static_cast<std::uint8_t>(data[pos++]);
        value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) {
            return value;
        }
        shift += 7;
    }
    throw std::runtime_error("Malformed protobuf varint");
}

void write_tag(std::string& out, std::uint32_t field, std::uint32_t wire) {
    write_varint(out, (static_cast<std::uint64_t>(field) << 3) | wire);
}

void write_int(std::string& out, std::uint32_t field, i64 value) {
    write_tag(out, field, kWireVarint);
    write_varint(out, static_cast<std::uint64_t>(value));
}

void write_bool(std::string& out, std::uint32_t field, bool value) {
    write_int(out, field, value ? 1 : 0);
}

void write_double(std::string& out, std::uint32_t field, double value) {
    write_tag(out, field, kWireFixed64);
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "Unexpected double size");
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
    }
}

double read_double(const std::string& data, std::size_t& pos) {
    if (pos + 8 > data.size()) {
        throw std::runtime_error("Malformed protobuf fixed64");
    }
    std::uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
        bits |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(data[pos++])) << (8 * i);
    }
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void write_message(std::string& out, std::uint32_t field, const std::string& message) {
    write_tag(out, field, kWireLengthDelimited);
    write_varint(out, message.size());
    out.append(message);
}

void write_packed_ints(std::string& out, std::uint32_t field, const std::vector<i64>& values) {
    std::string payload;
    for (i64 value : values) {
        write_varint(payload, static_cast<std::uint64_t>(value));
    }
    write_message(out, field, payload);
}

void skip_field(const std::string& data, std::size_t& pos, std::uint32_t wire) {
    if (wire == kWireVarint) {
        read_varint(data, pos);
    } else if (wire == kWireFixed64) {
        if (pos + 8 > data.size()) {
            throw std::runtime_error("Malformed protobuf fixed64");
        }
        pos += 8;
    } else if (wire == kWireLengthDelimited) {
        std::uint64_t len = read_varint(data, pos);
        if (pos + len > data.size()) {
            throw std::runtime_error("Malformed protobuf length-delimited field");
        }
        pos += len;
    } else {
        throw std::runtime_error("Unsupported protobuf wire type");
    }
}

std::vector<i64> read_packed_ints(const std::string& data, std::size_t& pos) {
    std::uint64_t len = read_varint(data, pos);
    if (pos + len > data.size()) {
        throw std::runtime_error("Malformed packed repeated field");
    }
    std::size_t end = pos + static_cast<std::size_t>(len);
    std::vector<i64> values;
    while (pos < end) {
        values.push_back(static_cast<i64>(read_varint(data, pos)));
    }
    return values;
}

}  // namespace proto

TransitionMapKind normalize_transition_map_kind(bool use_small_dict, const std::string& map_type) {
    return transition_map_kind_from_name(map_type, use_small_dict);
}

class SamBase {
public:
    SamBase(
        i64 n_predicts,
        double alpha,
        i64 k,
        bool use_gsam,
        bool use_small_dict,
        TransitionMapKind map_kind,
        i64 lazy_threshold)
        : n_predicts_(n_predicts),
          alpha_(alpha),
          k_(k),
          use_gsam_(use_gsam),
          use_small_dict_(map_kind == TransitionMapKind::kLazy),
          map_kind_(map_kind),
          lazy_threshold_(lazy_threshold) {
        (void)use_small_dict;
        reset_storage();
    }

    std::pair<i64, i64> lookup(i64 token) const {
        return transfer_state(cur_index_, cur_length_, token);
    }

    void transfer_tokens(const std::vector<i64>& tokens) {
        for (i64 token : tokens) {
            auto state = transfer_state(cur_index_, cur_length_, token);
            cur_index_ = state.first;
            cur_length_ = state.second;
        }
    }

    void reset_cursor() {
        cur_index_ = 0;
        cur_length_ = 0;
    }

    std::vector<i64> gen_draft(i64 index, i64 start_token) const {
        if (index < 0 || index >= static_cast<i64>(states_.size())) {
            index = 0;
        }
        i64 endpos = states_[index].min_endpos;
        std::vector<i64> pred_ids;
        pred_ids.reserve(static_cast<std::size_t>(n_predicts_));
        pred_ids.push_back(start_token);
        append_following_tokens(pred_ids, endpos, n_predicts_);
        while (static_cast<i64>(pred_ids.size()) < n_predicts_) {
            pred_ids.push_back(0);
        }
        return pred_ids;
    }

    i64 state_count() const {
        return static_cast<i64>(states_.size());
    }

    EdgeStats edge_stats() const {
        EdgeStats stats;
        stats.total_states = static_cast<i64>(states_.size());
        for (const State& state : states_) {
            if (state.next->size() == 1) {
                stats.single_next_states += 1;
            }
        }
        if (stats.total_states > 0) {
            stats.ratio = static_cast<double>(stats.single_next_states) /
                          static_cast<double>(stats.total_states);
        }
        return stats;
    }

    bool use_gsam() const {
        return use_gsam_;
    }

    bool use_small_dict() const {
        return use_small_dict_;
    }

    std::string map_type() const {
        return transition_map_kind_name(map_kind_);
    }

    i64 lazy_threshold() const {
        return lazy_threshold_;
    }

    i64 transition_memory_usage() const {
        std::size_t bytes = sizeof(*this) + states_.capacity() * sizeof(State);
        for (const State& state : states_) {
            bytes += state.next->memory_usage();
        }
        if (bytes > static_cast<std::size_t>(std::numeric_limits<i64>::max())) {
            throw std::overflow_error("transition memory usage does not fit in int64_t");
        }
        return static_cast<i64>(bytes);
    }

    i64 n_predicts() const {
        return n_predicts_;
    }

    void set_n_predicts(i64 value) {
        n_predicts_ = value;
    }

protected:
    void reset_storage() {
        states_.clear();
        states_.emplace_back(map_kind_, lazy_threshold_, -1, 0);
        input_ids_.clear();
        input_ids_.push_back(-1);
        boundary_end_.clear();
        boundary_end_.push_back(0);
        last_ = 0;
        cur_index_ = 0;
        cur_length_ = 0;
        max_length_ = 0;
    }

    i64 expand_state(const State& state) {
        states_.push_back(state);
        return static_cast<i64>(states_.size()) - 1;
    }

    i64 extend_standard(i64 token, i64 endpos) {
        max_length_ += 1;
        State cur_state(map_kind_, lazy_threshold_, -1, max_length_);
        cur_state.min_endpos = endpos;
        i64 cur = expand_state(cur_state);
        i64 p = last_;
        while (p != -1 && !states_[p].next->contains(token)) {
            states_[p].next->set(token, cur);
            p = states_[p].link;
        }
        if (p == -1) {
            states_[cur].link = 0;
        } else {
            i64 q = states_[p].next->get(token);
            if (states_[p].length + 1 == states_[q].length) {
                states_[cur].link = q;
            } else {
                State clone_state = states_[q];
                i64 clone = expand_state(clone_state);
                states_[clone].length = states_[p].length + 1;
                states_[clone].cnt_endpos = 0;
                while (p != -1 && states_[p].next->get(token) == q) {
                    states_[p].next->set(token, clone);
                    p = states_[p].link;
                }
                states_[q].link = clone;
                states_[cur].link = clone;
            }
        }
        last_ = cur;
        states_[cur].cnt_endpos += 1;
        return cur;
    }

    i64 extend_gsam(i64 last, i64 token, i64 endpos) {
        i64 existing = states_[last].next->get(token);
        if (existing != -1 && states_[existing].length == states_[last].length + 1) {
            states_[existing].cnt_endpos += 1;
            return existing;
        }

        bool created_cur = true;
        State cur_state(map_kind_, lazy_threshold_, -1, states_[last].length + 1);
        cur_state.min_endpos = endpos;
        i64 cur = expand_state(cur_state);
        i64 p = last;
        while (p != -1 && !states_[p].next->contains(token)) {
            states_[p].next->set(token, cur);
            p = states_[p].link;
        }
        if (p == -1) {
            states_[cur].link = 0;
            states_[cur].cnt_endpos += 1;
            return cur;
        }
        i64 q = states_[p].next->get(token);
        if (states_[p].length + 1 == states_[q].length) {
            states_[cur].link = q;
            states_[cur].cnt_endpos += 1;
            return cur;
        }
        if (p == last) {
            created_cur = false;
            states_.pop_back();
        }

        State clone_state = states_[q];
        i64 clone = expand_state(clone_state);
        states_[clone].length = states_[p].length + 1;
        states_[clone].cnt_endpos = 0;
        while (p != -1 && states_[p].next->get(token) == q) {
            states_[p].next->set(token, clone);
            p = states_[p].link;
        }
        states_[q].link = clone;
        if (created_cur) {
            states_[cur].link = clone;
            states_[cur].cnt_endpos += 1;
            return cur;
        }
        states_[clone].cnt_endpos += 1;
        return clone;
    }

    std::pair<i64, i64> transfer_state(i64 index, i64 length, i64 token) const {
        while (index != 0 && !states_[index].next->contains(token)) {
            index = states_[index].link;
            length = states_[index].length;
        }
        if (states_[index].next->contains(token)) {
            index = states_[index].next->get(token);
            length += 1;
        } else {
            index = 0;
            length = 0;
        }
        return {index, length};
    }

    void propagate_counts() {
        std::vector<i64> order(states_.size());
        for (i64 i = 0; i < static_cast<i64>(states_.size()); ++i) {
            order[static_cast<std::size_t>(i)] = i;
        }
        std::sort(order.begin(), order.end(), [this](i64 lhs, i64 rhs) {
            return states_[lhs].length > states_[rhs].length;
        });
        for (i64 index : order) {
            i64 link = states_[index].link;
            if (index != 0 && link >= 0) {
                states_[link].cnt_endpos += states_[index].cnt_endpos;
                states_[link].min_endpos = std::min(states_[link].min_endpos, states_[index].min_endpos);
            }
        }
    }

    void append_following_tokens(std::vector<i64>& out, i64 endpos, i64 limit) const {
        if (endpos < 0 || endpos >= static_cast<i64>(input_ids_.size())) {
            return;
        }
        i64 boundary = boundary_end_[static_cast<std::size_t>(endpos)];
        for (i64 pos = endpos + 1; pos <= boundary && static_cast<i64>(out.size()) < limit; ++pos) {
            out.push_back(input_ids_[static_cast<std::size_t>(pos)]);
        }
    }

    i64 n_predicts_ = 40;
    double alpha_ = 4.0;
    i64 k_ = 8;
    bool use_gsam_ = true;
    bool use_small_dict_ = true;
    TransitionMapKind map_kind_ = TransitionMapKind::kLazy;
    i64 lazy_threshold_ = 1;
    std::vector<State> states_;
    std::vector<i64> input_ids_;
    std::vector<i64> boundary_end_;
    std::vector<std::vector<std::pair<i64, i64>>> states_topk_next_;
    i64 last_ = 0;
    i64 cur_index_ = 0;
    i64 cur_length_ = 0;
    i64 max_length_ = 0;
};

class DynSAMCore : public SamBase {
public:
    DynSAMCore(
        i64 n_predicts = 40,
        double alpha = 4.0,
        bool use_gsam = true,
        bool use_small_dict = true,
        const std::string& map_type = "",
        i64 lazy_threshold = 1)
        : SamBase(
              n_predicts,
              alpha,
              8,
              use_gsam,
              use_small_dict,
              normalize_transition_map_kind(use_small_dict, map_type),
              lazy_threshold) {
        print_config();
    }

    void reset() {
        reset_storage();
    }

    void add_tokens(const std::vector<i64>& tokens) {
        for (i64 token : tokens) {
            input_ids_.push_back(token);
            boundary_end_.push_back(static_cast<i64>(input_ids_.size()) - 1);
            auto state = transfer_state(cur_index_, cur_length_, token);
            cur_index_ = state.first;
            cur_length_ = state.second;
            extend_standard(token, static_cast<i64>(input_ids_.size()) - 1);
        }
        i64 current_end = static_cast<i64>(input_ids_.size()) - 1;
        for (std::size_t i = 1; i < boundary_end_.size(); ++i) {
            boundary_end_[i] = current_end;
        }
    }

    DynDraftResult gen_dyn_draft(i64 index, i64 match_length, i64 start_token) const {
        i64 n = std::min(n_predicts_, static_cast<i64>(1 + static_cast<i64>(match_length * alpha_)));
        std::vector<i64> seq;
        seq.reserve(static_cast<std::size_t>(std::max<i64>(n, 1)));
        seq.push_back(start_token);
        if (index >= 0 && index < static_cast<i64>(states_.size())) {
            append_following_tokens(seq, states_[index].min_endpos, n);
        }
        return {seq, {}};
    }

private:
    void print_config() const {
        py::print(
            "GSAMD dynamic SAM config: use_gsam=",
            use_gsam_,
            ", map_type=",
            map_type(),
            ", lazy_threshold=",
            lazy_threshold_,
            py::arg("sep") = "");
    }
};

class StaticSAMCore : public SamBase {
public:
    StaticSAMCore(
        i64 n_predicts = 40,
        double alpha = 4.0,
        i64 k = 8,
        bool use_gsam = true,
        bool use_small_dict = true,
        const std::string& map_type = "",
        i64 lazy_threshold = 1)
        : SamBase(
              n_predicts,
              alpha,
              k,
              use_gsam,
              use_small_dict,
              normalize_transition_map_kind(use_small_dict, map_type),
              lazy_threshold) {}

    void add_batch_tokens(const std::vector<std::vector<i64>>& batch_tokens, i64 eos_token, bool add_eos = true) {
        reset_storage();
        for (const auto& source_tokens : batch_tokens) {
            std::vector<i64> tokens = source_tokens;
            if (!use_gsam_ && add_eos && (tokens.empty() || tokens.back() != eos_token)) {
                tokens.push_back(eos_token);
            }
            add_sequence(tokens);
        }
        propagate_counts();
    }

    void add_sequence(const std::vector<i64>& tokens) {
        if (tokens.empty()) {
            return;
        }
        i64 sequence_start = static_cast<i64>(input_ids_.size());
        i64 sequence_end = sequence_start + static_cast<i64>(tokens.size()) - 1;
        i64 local_last = 0;
        for (i64 token : tokens) {
            input_ids_.push_back(token);
            boundary_end_.push_back(sequence_end);
            i64 endpos = static_cast<i64>(input_ids_.size()) - 1;
            if (use_gsam_) {
                local_last = extend_gsam(local_last, token, endpos);
            } else {
                extend_standard(token, endpos);
            }
        }
    }

    void init_topk_next() {
        states_topk_next_.assign(states_.size(), {});
        for (i64 index = 0; index < static_cast<i64>(states_.size()); ++index) {
            auto edges = states_[index].next->items();
            std::sort(edges.begin(), edges.end(), [this](const auto& lhs, const auto& rhs) {
                i64 lhs_count = states_[lhs.second].cnt_endpos;
                i64 rhs_count = states_[rhs.second].cnt_endpos;
                if (lhs_count != rhs_count) {
                    return lhs_count > rhs_count;
                }
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            });
            if (static_cast<i64>(edges.size()) > k_) {
                edges.resize(static_cast<std::size_t>(k_));
            }
            states_topk_next_[static_cast<std::size_t>(index)] = std::move(edges);
        }
    }

    DynDraftResult gen_dyn_draft(i64 index, i64 match_length, i64 start_token) const {
        struct Item {
            double prob;
            i64 token;
            i64 index;
            i64 ancestor;
            i64 depth;
        };
        struct Compare {
            bool operator()(const Item& lhs, const Item& rhs) const {
                return lhs.prob > rhs.prob;
            }
        };

        i64 n = std::min(n_predicts_, static_cast<i64>(1 + static_cast<i64>(match_length * alpha_)));
        std::priority_queue<Item, std::vector<Item>, Compare> heap;
        std::vector<i64> tree;
        std::vector<i64> ancestors;
        std::unordered_map<i64, i64> depth_counts;
        heap.push({-1.0, start_token, index, -1, 0});
        while (static_cast<i64>(tree.size()) != n && !heap.empty()) {
            Item item = heap.top();
            heap.pop();
            if (depth_counts[item.depth] + 1 > k_) {
                continue;
            }
            depth_counts[item.depth] += 1;
            i64 tree_index = static_cast<i64>(tree.size());
            tree.push_back(item.token);
            ancestors.push_back(item.ancestor);
            if (static_cast<i64>(tree.size()) == n) {
                break;
            }
            if (item.index < 0 || item.index >= static_cast<i64>(states_topk_next_.size())) {
                continue;
            }
            i64 cnt_sum = states_[item.index].cnt_endpos;
            if (cnt_sum <= 0) {
                continue;
            }
            const auto& next_states = states_topk_next_[static_cast<std::size_t>(item.index)];
            for (const auto& edge : next_states) {
                double next_prob = static_cast<double>(states_[edge.second].cnt_endpos) /
                                   static_cast<double>(cnt_sum);
                heap.push({item.prob * next_prob, edge.first, edge.second, tree_index, item.depth + 1});
            }
        }
        return {tree, ancestors};
    }

    void save(const std::string& path) const {
        std::string root;
        proto::write_int(root, 1, proto::kVersion);
        proto::write_bool(root, 2, use_gsam_);
        proto::write_bool(root, 3, use_small_dict_);
        proto::write_int(root, 4, n_predicts_);
        proto::write_double(root, 5, alpha_);
        proto::write_int(root, 6, k_);
        proto::write_packed_ints(root, 7, input_ids_);
        proto::write_packed_ints(root, 8, boundary_end_);
        for (i64 index = 0; index < static_cast<i64>(states_.size()); ++index) {
            proto::write_message(root, 9, encode_state(index));
        }
        proto::write_int(root, 10, static_cast<i64>(map_kind_));
        proto::write_int(root, 11, lazy_threshold_);

        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open SAM file for writing: " + path);
        }
        out.write(root.data(), static_cast<std::streamsize>(root.size()));
    }

    static StaticSAMCore load(
        const std::string& path,
        const std::string& map_type = "",
        i64 lazy_threshold_override = 1) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open SAM file for reading: " + path);
        }
        std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

        std::uint32_t version = 0;
        bool use_gsam = true;
        bool use_small_dict = true;
        i64 n_predicts = 40;
        double alpha = 4.0;
        i64 k = 8;
        bool saw_map_kind = false;
        TransitionMapKind map_kind = TransitionMapKind::kLazy;
        i64 lazy_threshold = 1;
        std::vector<i64> input_ids;
        std::vector<i64> boundary_end;
        std::vector<std::string> encoded_states;

        std::size_t pos = 0;
        while (pos < data.size()) {
            std::uint64_t tag = proto::read_varint(data, pos);
            std::uint32_t field = static_cast<std::uint32_t>(tag >> 3);
            std::uint32_t wire = static_cast<std::uint32_t>(tag & 0x7);
            if (field == 1) {
                version = static_cast<std::uint32_t>(proto::read_varint(data, pos));
            } else if (field == 2) {
                use_gsam = proto::read_varint(data, pos) != 0;
            } else if (field == 3) {
                use_small_dict = proto::read_varint(data, pos) != 0;
            } else if (field == 4) {
                n_predicts = static_cast<i64>(proto::read_varint(data, pos));
            } else if (field == 5) {
                alpha = proto::read_double(data, pos);
            } else if (field == 6) {
                k = static_cast<i64>(proto::read_varint(data, pos));
            } else if (field == 7 && wire == proto::kWireLengthDelimited) {
                input_ids = proto::read_packed_ints(data, pos);
            } else if (field == 8 && wire == proto::kWireLengthDelimited) {
                boundary_end = proto::read_packed_ints(data, pos);
            } else if (field == 9 && wire == proto::kWireLengthDelimited) {
                std::uint64_t len = proto::read_varint(data, pos);
                if (pos + len > data.size()) {
                    throw std::runtime_error("Malformed state message");
                }
                encoded_states.emplace_back(data.substr(pos, static_cast<std::size_t>(len)));
                pos += static_cast<std::size_t>(len);
            } else if (field == 10) {
                map_kind = static_cast<TransitionMapKind>(static_cast<i32>(proto::read_varint(data, pos)));
                saw_map_kind = true;
            } else if (field == 11) {
                lazy_threshold = static_cast<i64>(proto::read_varint(data, pos));
            } else {
                proto::skip_field(data, pos, wire);
            }
        }

        if (version != 1 && version != proto::kVersion) {
            throw std::runtime_error("Unsupported GSAMD SAM file version");
        }
        if (!map_type.empty()) {
            map_kind = transition_map_kind_from_name(map_type, true);
            lazy_threshold = lazy_threshold_override;
        } else if (!saw_map_kind) {
            map_kind = use_small_dict ? TransitionMapKind::kLazy : TransitionMapKind::kUnordered;
        }
        StaticSAMCore sam(
            n_predicts,
            alpha,
            k,
            use_gsam,
            map_kind == TransitionMapKind::kLazy,
            transition_map_kind_name(map_kind),
            lazy_threshold);
        sam.input_ids_ = std::move(input_ids);
        sam.boundary_end_ = std::move(boundary_end);
        sam.states_.clear();
        sam.states_.reserve(encoded_states.size());
        sam.states_topk_next_.clear();
        sam.states_topk_next_.reserve(encoded_states.size());
        for (const std::string& encoded : encoded_states) {
            sam.decode_state(encoded);
        }
        sam.reset_cursor();
        py::print(
            "GSAMD static SAM config: use_gsam=",
            sam.use_gsam_,
            ", map_type=",
            sam.map_type(),
            ", lazy_threshold=",
            sam.lazy_threshold_,
            py::arg("sep") = "");
        return sam;
    }

private:
    std::string encode_edge(i64 token, i64 state) const {
        std::string out;
        proto::write_int(out, 1, token);
        proto::write_int(out, 2, state);
        return out;
    }

    std::string encode_state(i64 index) const {
        const State& state = states_[static_cast<std::size_t>(index)];
        std::string out;
        proto::write_int(out, 1, state.link);
        proto::write_int(out, 2, state.length);
        proto::write_int(out, 3, state.cnt_endpos);
        proto::write_int(out, 4, state.min_endpos);
        auto edges = state.next->items();
        std::sort(edges.begin(), edges.end());
        for (const auto& edge : edges) {
            proto::write_message(out, 5, encode_edge(edge.first, edge.second));
        }
        if (index < static_cast<i64>(states_topk_next_.size())) {
            auto topk_edges = states_topk_next_[static_cast<std::size_t>(index)];
            std::sort(topk_edges.begin(), topk_edges.end(), [this](const auto& lhs, const auto& rhs) {
                i64 lhs_count = states_[lhs.second].cnt_endpos;
                i64 rhs_count = states_[rhs.second].cnt_endpos;
                if (lhs_count != rhs_count) {
                    return lhs_count > rhs_count;
                }
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            });
            for (const auto& edge : topk_edges) {
                proto::write_message(out, 6, encode_edge(edge.first, edge.second));
            }
        }
        return out;
    }

    std::pair<i64, i64> decode_edge(const std::string& data) const {
        i64 token = 0;
        i64 state = 0;
        std::size_t pos = 0;
        while (pos < data.size()) {
            std::uint64_t tag = proto::read_varint(data, pos);
            std::uint32_t field = static_cast<std::uint32_t>(tag >> 3);
            std::uint32_t wire = static_cast<std::uint32_t>(tag & 0x7);
            if (field == 1) {
                token = static_cast<i64>(proto::read_varint(data, pos));
            } else if (field == 2) {
                state = static_cast<i64>(proto::read_varint(data, pos));
            } else {
                proto::skip_field(data, pos, wire);
            }
        }
        return {token, state};
    }

    void decode_state(const std::string& data) {
        State state(map_kind_, lazy_threshold_);
        std::vector<std::pair<i64, i64>> topk;
        std::size_t pos = 0;
        while (pos < data.size()) {
            std::uint64_t tag = proto::read_varint(data, pos);
            std::uint32_t field = static_cast<std::uint32_t>(tag >> 3);
            std::uint32_t wire = static_cast<std::uint32_t>(tag & 0x7);
            if (field == 1) {
                state.link = static_cast<i64>(proto::read_varint(data, pos));
            } else if (field == 2) {
                state.length = static_cast<i64>(proto::read_varint(data, pos));
            } else if (field == 3) {
                state.cnt_endpos = static_cast<i64>(proto::read_varint(data, pos));
            } else if (field == 4) {
                state.min_endpos = static_cast<i64>(proto::read_varint(data, pos));
            } else if ((field == 5 || field == 6) && wire == proto::kWireLengthDelimited) {
                std::uint64_t len = proto::read_varint(data, pos);
                if (pos + len > data.size()) {
                    throw std::runtime_error("Malformed edge message");
                }
                auto edge = decode_edge(data.substr(pos, static_cast<std::size_t>(len)));
                pos += static_cast<std::size_t>(len);
                if (field == 5) {
                    state.next->set(edge.first, edge.second);
                } else {
                    topk.push_back(edge);
                }
            } else {
                proto::skip_field(data, pos, wire);
            }
        }
        states_.push_back(state);
        states_topk_next_.push_back(std::move(topk));
    }
};

PYBIND11_MODULE(_gsamd_core, m) {
    py::class_<DynDraftResult>(m, "DynDraftResult")
        .def_readonly("tokens", &DynDraftResult::tokens)
        .def_readonly("ancestors", &DynDraftResult::ancestors);

    py::class_<EdgeStats>(m, "EdgeStats")
        .def_readonly("single_next_states", &EdgeStats::single_next_states)
        .def_readonly("total_states", &EdgeStats::total_states)
        .def_readonly("ratio", &EdgeStats::ratio);

    py::class_<DynSAMCore>(m, "DynSAMCore")
        .def(py::init<i64, double, bool, bool, const std::string&, i64>(),
             py::arg("n_predicts") = 40,
             py::arg("alpha") = 4.0,
             py::arg("use_gsam") = true,
             py::arg("use_small_dict") = true,
             py::arg("map_type") = "",
             py::arg("lazy_threshold") = 1)
        .def("reset", &DynSAMCore::reset)
        .def("add_tokens", &DynSAMCore::add_tokens)
        .def("transfer_tokens", &DynSAMCore::transfer_tokens)
        .def("lookup", &DynSAMCore::lookup)
        .def("gen_draft", &DynSAMCore::gen_draft)
        .def("gen_dyn_draft", &DynSAMCore::gen_dyn_draft)
        .def("state_count", &DynSAMCore::state_count)
        .def("edge_stats", &DynSAMCore::edge_stats)
        .def("transition_memory_usage", &DynSAMCore::transition_memory_usage)
        .def_property("n_predicts", &DynSAMCore::n_predicts, &DynSAMCore::set_n_predicts)
        .def_property_readonly("use_gsam", &DynSAMCore::use_gsam)
        .def_property_readonly("use_small_dict", &DynSAMCore::use_small_dict)
        .def_property_readonly("map_type", &DynSAMCore::map_type)
        .def_property_readonly("lazy_threshold", &DynSAMCore::lazy_threshold);

    py::class_<StaticSAMCore>(m, "StaticSAMCore")
        .def(py::init<i64, double, i64, bool, bool, const std::string&, i64>(),
             py::arg("n_predicts") = 40,
             py::arg("alpha") = 4.0,
             py::arg("K") = 8,
             py::arg("use_gsam") = true,
             py::arg("use_small_dict") = true,
             py::arg("map_type") = "",
             py::arg("lazy_threshold") = 1)
        .def("add_batch_tokens", &StaticSAMCore::add_batch_tokens,
             py::arg("batch_tokens"),
             py::arg("eos_token"),
             py::arg("add_eos") = true)
        .def("init_topk_next", &StaticSAMCore::init_topk_next)
        .def("transfer_tokens", &StaticSAMCore::transfer_tokens)
        .def("lookup", &StaticSAMCore::lookup)
        .def("reset", &StaticSAMCore::reset_cursor)
        .def("gen_draft", &StaticSAMCore::gen_draft)
        .def("gen_dyn_draft", &StaticSAMCore::gen_dyn_draft)
        .def("save", &StaticSAMCore::save)
        .def_static(
            "load",
            &StaticSAMCore::load,
            py::arg("path"),
            py::arg("map_type") = "",
            py::arg("lazy_threshold") = 1)
        .def("state_count", &StaticSAMCore::state_count)
        .def("edge_stats", &StaticSAMCore::edge_stats)
        .def("transition_memory_usage", &StaticSAMCore::transition_memory_usage)
        .def_property("n_predicts", &StaticSAMCore::n_predicts, &StaticSAMCore::set_n_predicts)
        .def_property_readonly("use_gsam", &StaticSAMCore::use_gsam)
        .def_property_readonly("use_small_dict", &StaticSAMCore::use_small_dict)
        .def_property_readonly("map_type", &StaticSAMCore::map_type)
        .def_property_readonly("lazy_threshold", &StaticSAMCore::lazy_threshold);
}
