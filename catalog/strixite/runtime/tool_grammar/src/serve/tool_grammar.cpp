#include "serve/tool_grammar.hpp"

#include "common/check.hpp"

#include <array>
#include <map>
#include <set>
#include <vector>

namespace strix::tool_grammar {

namespace {

using grammar::Alternative;
using grammar::ByteRange;
using grammar::Element;
using grammar::Grammar;

Alternative ref(int rule) { return Alternative{Element::reference(rule)}; }
Alternative literal(std::string_view bytes) { return Grammar::literal(bytes); }
Alternative bytes(std::vector<ByteRange> ranges) { return Alternative{Element::byte_set(std::move(ranges))}; }
Alternative sequence(std::initializer_list<Alternative> parts) {
    Alternative out;
    for (const Alternative &part : parts) out.insert(out.end(), part.begin(), part.end());
    return out;
}

// Every byte value except those in `except`, as merged ranges.
std::vector<ByteRange> all_but(const std::set<uint8_t> &except) {
    std::vector<ByteRange> out;
    for (int b = 0; b < 256; ++b) {
        if (except.count((uint8_t)b)) continue;
        if (!out.empty() && out.back().high + 1 == b) out.back().high = (uint8_t)b;
        else out.push_back({(uint8_t)b, (uint8_t)b});
    }
    return out;
}

bool tag_safe(const std::string &name) {
    return !name.empty() && name.find('>') == std::string::npos && name.find('\n') == std::string::npos;
}

class Builder {
public:
    // Up to kMaxWhitespaceRun bytes of white space (space, tab, newline, carriage return), possibly none.
    int whitespace() {
        if (ws_ >= 0) return ws_;
        std::vector<int> rules(kMaxWhitespaceRun + 1);
        for (int k = 0; k <= kMaxWhitespaceRun; ++k) rules[k] = g_.add_rule("ws." + std::to_string(k));
        for (int k = 0; k <= kMaxWhitespaceRun; ++k) {
            g_.add_alternative(rules[k], {});
            if (k < kMaxWhitespaceRun)
                g_.add_alternative(rules[k], sequence({bytes({{' ', ' '}, {'\t', '\t'}, {'\n', '\n'}, {'\r', '\r'}}),
                                                       ref(rules[k + 1])}));
        }
        return ws_ = rules[0];
    }

    // Any byte string (empty included) that contains none of `needles`: one rule per state of the needles' Aho-Corasick
    // automaton (a trie node with failure links), each `empty | byte -> next state`, the transitions that would
    // complete a needle left out. Right-recursive, so the matcher's stack doesn't grow with the text.
    int not_containing(const std::string &name, const std::vector<std::string> &needles) {
        STRIX_CHECK(!needles.empty(), "tool grammar: no needles");
        // The trie: node 0 is the root; next[node][byte] = child or -1; terminal = a needle ends here (or via a link).
        std::vector<std::array<int, 256>> next(1);
        std::vector<bool> terminal(1, false);
        next[0].fill(-1);
        for (const std::string &needle : needles) {
            STRIX_CHECK(!needle.empty(), "tool grammar: an empty needle");
            int node = 0;
            for (unsigned char c : needle) {
                if (next[node][c] < 0) {
                    next[node][c] = (int)next.size();
                    next.emplace_back().fill(-1);
                    terminal.push_back(false);
                }
                node = next[node][c];
            }
            terminal[node] = true;
        }
        // Failure links by breadth-first order; goto() completes the transition table (missing edge -> the link's).
        std::vector<int> fail(next.size(), 0), order;
        for (int c = 0; c < 256; ++c)
            if (next[0][c] >= 0) order.push_back(next[0][c]); else next[0][c] = 0;
        for (size_t i = 0; i < order.size(); ++i) {
            const int node = order[i];
            terminal[node] = terminal[node] || terminal[fail[node]];
            for (int c = 0; c < 256; ++c) {
                const int child = next[node][c];
                if (child >= 0) {
                    fail[child] = next[fail[node]][c];
                    order.push_back(child);
                } else {
                    next[node][c] = next[fail[node]][c];
                }
            }
        }
        std::vector<int> rules(next.size(), -1);
        for (size_t k = 0; k < next.size(); ++k)
            if (!terminal[k]) rules[k] = g_.add_rule(name + "." + std::to_string(k));
        for (size_t k = 0; k < next.size(); ++k) {
            if (terminal[k]) continue;
            g_.add_alternative(rules[k], {});
            std::map<int, std::vector<ByteRange>> by_next;
            for (int b = 0; b < 256; ++b) {
                const int j = next[k][b];
                if (terminal[j]) continue;  // this byte would complete a needle
                auto &ranges = by_next[j];
                if (!ranges.empty() && ranges.back().high + 1 == b) ranges.back().high = (uint8_t)b;
                else ranges.push_back({(uint8_t)b, (uint8_t)b});
            }
            for (auto &[j, ranges] : by_next) g_.add_alternative(rules[k], sequence({bytes(std::move(ranges)), ref(rules[j])}));
        }
        return rules[0];
    }

    // A parameter name when the tool declares no properties: one or more bytes, none '>' or newline (what the
    // parser accepts).
    int any_param_name() {
        if (any_name_ >= 0) return any_name_;
        const int rule = g_.add_rule("param.any");
        const auto ch = bytes(all_but({'>', '\n', 0}));
        g_.add_alternative(rule, ch);
        g_.add_alternative(rule, sequence({ch, ref(rule)}));
        return any_name_ = rule;
    }

    int value() {
        if (value_ >= 0) return value_;
        // A value may hold other tags as text, but not the call tags the parser reads as tokens, nor its own end.
        return value_ = not_containing("value", {"</parameter>", "<tool_call>", "</tool_call>", std::string(kCallBegin), std::string(kCallEnd)});
    }

    // The call rule of one tool: `<tool_call>` ws `<function=NAME>` ws params `</function>` ws newline `</tool_call>`.
    int call(const std::string &name, const std::vector<std::string> &properties, size_t index) {
        const std::string tag = "tool." + std::to_string(index);
        const int params = g_.add_rule(tag + ".params");
        const int param = g_.add_rule(tag + ".param");
        g_.add_alternative(params, {});
        g_.add_alternative(params, sequence({ref(param), ref(params)}));
        const int pname = g_.add_rule(tag + ".param.name");
        if (properties.empty()) g_.add_alternative(pname, ref(any_param_name()));
        for (const std::string &p : properties) g_.add_alternative(pname, literal(p));
        g_.add_alternative(param, sequence({literal("<parameter="), ref(pname), literal(">"), ref(value()),
                                            literal("</parameter>"), ref(whitespace())}));
        const int rule = g_.add_rule(tag + ".call");
        g_.add_alternative(rule, sequence({literal(kCallBegin), ref(whitespace()), literal("<function=" + name + ">"),
                                           ref(whitespace()), ref(params), literal("</function>"), ref(whitespace()),
                                           literal("\n"), literal(kCallEnd)}));
        return rule;
    }

    std::shared_ptr<const Grammar> finish(const std::vector<int> &calls, bool required) {
        const int call = g_.add_rule("call");
        for (int c : calls) g_.add_alternative(call, ref(c));
        // more := empty | ws newline call more   (a later call stands right after a newline)
        const int more = g_.add_rule("more");
        g_.add_alternative(more, {});
        g_.add_alternative(more, sequence({ref(whitespace()), literal("\n"), ref(call), ref(more)}));
        // Free text holds no call tag: not the tag tokens (their sentinels), not their text spelling, and not a bare
        // `<function=`, which is a call the parser would read as text (seen from a steered model).
        const int text = not_containing("text", {"<tool_call>", "</tool_call>", "<function=", std::string(kCallBegin), std::string(kCallEnd)});
        // text_nl := empty | text newline   (the first call stands at the start or right after a newline)
        const int text_nl = g_.add_rule("text_nl");
        g_.add_alternative(text_nl, {});
        g_.add_alternative(text_nl, sequence({ref(text), literal("\n")}));
        const int root = g_.add_rule("root");
        if (required) {
            // tool_choice required / a name: the answer is the call(s); free text would let a model under pressure
            // talk until the length cap without ever calling (seen at steering scale 6). Reasoning has the think block.
            g_.add_alternative(root, sequence({ref(call), ref(more)}));
        } else {
            g_.add_alternative(root, ref(text));
            g_.add_alternative(root, sequence({ref(text_nl), ref(call), ref(more)}));
        }
        g_.set_root(root);
        g_.validate();
        return std::make_shared<const Grammar>(std::move(g_));
    }

private:
    Grammar g_;
    int ws_ = -1, any_name_ = -1, value_ = -1;
};

}  // namespace

std::shared_ptr<const Grammar> compile(const json::Value &tools, const std::string &choice) {
    STRIX_CHECK(tools.is_array(), "tool grammar: 'tools' must be an array");
    const auto &ts = tools.as_array("tools");
    STRIX_CHECK(!ts.empty(), "tool grammar: no tools");
    Builder b;
    std::vector<int> calls;
    bool named = false;
    for (size_t k = 0; k < ts.size(); ++k) {
        const std::string p = "tools[" + std::to_string(k) + "]";
        const json::Value *fn = ts[k].find("function");
        STRIX_CHECK(fn && fn->is_object(), "tool grammar: '", p, ".function' must be an object");
        const json::Value *name = fn->find("name");
        STRIX_CHECK(name && name->is_string(), "tool grammar: '", p, ".function.name' must be a string");
        const std::string &nm = name->as_string("name");
        STRIX_CHECK(tag_safe(nm), "tool grammar: '", p, ".function.name' cannot be written as <function=", nm,
                    "> (empty, or holds '>' or a newline)");
        std::vector<std::string> properties;
        if (const json::Value *params = fn->find("parameters"); params && params->is_object())
            if (const json::Value *props = params->find("properties"); props && props->is_object())
                for (const auto &[pn, _] : props->as_object("properties")) {
                    STRIX_CHECK(tag_safe(pn), "tool grammar: '", p, ".function.parameters.properties' key '", pn,
                                "' cannot be written as <parameter=", pn, ">");
                    properties.push_back(pn);
                }
        if (choice != "auto" && choice != "required") {
            if (nm != choice) continue;
            named = true;
        }
        calls.push_back(b.call(nm, properties, k));
    }
    if (choice != "auto" && choice != "required")
        STRIX_CHECK(named, "tool grammar: tool_choice names '", choice, "', which is not among the tools");
    return b.finish(calls, choice != "auto");
}

}  // namespace strix::tool_grammar
