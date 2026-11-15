// The tool-call grammar (serve/tool_grammar.hpp) against the matcher, host only (no model, no GPU): well-formed
// calls and plain text are accepted, the malformed shapes that bit a steered agent are refused, tool_choice
// required / named behave, every call the grammar accepts parses, and, given the tokenizer, the token masks under
// StructuredOutput let exactly the tag tokens open and close a call. Exit 0 when every case holds.
//
//   tool_grammar_check [tokenizer.json]

#include "common/check.hpp"
#include "serve/grammar.hpp"
#include "serve/json.hpp"
#include "serve/output_parser.hpp"
#include "serve/structured_output.hpp"
#include "serve/token_mask.hpp"
#include "serve/tokenizer.hpp"
#include "serve/tool_grammar.hpp"

#include <cstdio>
#include <string>

using namespace strix;

namespace {

int failures = 0;

// Whether `text` is a complete sentence of the grammar (every byte accepted and the end allowed there).
bool accepts(const std::shared_ptr<const grammar::Grammar> &g, const std::string &text) {
    grammar::GrammarMatcher m(g);
    return m.advance(text) && m.accepting();
}
// Whether every byte of `text` is accepted (the end may or may not be allowed).
bool allows(const std::shared_ptr<const grammar::Grammar> &g, const std::string &text) {
    grammar::GrammarMatcher m(g);
    return m.advance(text);
}

void expect(bool ok, const char *what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

const char *kTools = R"([
 {"type":"function","function":{"name":"run","description":"shell","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout_s":{"type":"integer"}},"required":["command"]}}},
 {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
 {"type":"function","function":{"name":"experts","parameters":{"type":"object"}}}
])";

// Raw-grammar cases spell the call tags as the grammar sees them: the sentinels StructuredOutput feeds for the tag
// tokens. The tokenizer cases use the tags' text, which the tokenizer turns into the tag tokens.
const std::string TC(tool_grammar::kCallBegin), TCE(tool_grammar::kCallEnd);
const std::string kBody = "\n<function=run>\n<parameter=command>\nls -la /data\n</parameter>\n</function>\n";
const std::string kCall = TC + kBody + TCE;
const std::string kCallText = "<tool_call>" + kBody + "</tool_call>";

}  // namespace

int main(int argc, char **argv) {
    const json::Value tools = json::Value::parse(kTools);
    const auto g = tool_grammar::compile(tools, "auto");
    std::printf("grammar: %zu rules\n", g->rules().size());

    expect(accepts(g, ""), "auto: an empty answer");
    expect(accepts(g, "Here is the reading list.\n1. x\n2. y"), "auto: plain text");
    expect(accepts(g, "Here is text with a <tool call> mention and a <function run> look-alike, and </parameter> too."), "auto: text with look-alike tags");
    expect(accepts(g, kCall), "auto: one call");
    expect(accepts(g, "Checking the shelf.\n\n" + kCall), "auto: text, blank line, call");
    expect(accepts(g, kCall + "\n" + kCall), "auto: two calls on consecutive lines");
    expect(accepts(g, kCall + "\n\n" + TC + "\n<function=experts>\n</function>\n" + TCE), "auto: a call of a tool with no properties and no parameters");
    expect(accepts(g, TC + "\n<function=experts>\n<parameter=anything>\nx\n</parameter>\n</function>\n" + TCE), "auto: a tool without properties takes any parameter name");
    expect(accepts(g, TC + "\n<function=run>\n<parameter=command>\necho '</function>'\n</parameter>\n<parameter=timeout_s>\n5\n</parameter>\n</function>\n" + TCE),
           "auto: a value may hold other tags; two parameters");
    expect(accepts(g, TC + "\n<function=run>\n<parameter=command>ls</parameter>\n</function>\n" + TCE), "auto: a value without the format's newlines");
    expect(accepts(g, TC + "\n<function=run>\n\n<parameter=command>\nls\n</parameter>\n\n</function>\n\n" + TCE), "auto: blank lines between tags");

    // Shapes seen from steered models: no such tool, a value without its tag, a bare function block, tags spelled
    // with ordinary tokens instead of the tag tokens.
    expect(!allows(g, TC + "\n<function=expert>\nlibrarian-brief\n</parameter>\n<parameter=question>\nTask\n</parameter>\n</function>\n" + TCE),
           "refuses the lib-3-brief shape (undeclared tool 'expert')");
    expect(!allows(g, TC + "\n<function=run>\nls\n</parameter>\n</function>\n" + TCE), "refuses a value without its <parameter=> tag");
    expect(!allows(g, TC + "\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>\n" + TCE), "refuses an undeclared parameter name");
    expect(!allows(g, TC + "\n<function=ru>\n</function>\n" + TCE), "refuses a prefix of a tool name");
    expect(!allows(g, "To call experts:\n<function=experts>\n<parameter=_raw>\nx\n</parameter>\n</function>"), "refuses a bare <function=> block outside a call (scale-8 replay shape)");
    expect(!allows(g, "Done.\n</tool_call>"), "refuses the text spelling of </tool_call> in text");
    expect(!allows(g, "Done.\n" + TCE), "refuses a stray </tool_call> token in text");
    expect(!allows(g, "Done.\n<tool_call>\n<function=run>\n</function>\n" + TCE), "refuses a call opened with the text spelling of <tool_call>");
    expect(!allows(g, TC + kBody + "</tool_call>"), "refuses a call closed with the text spelling of </tool_call> (the scale-8 shape)");
    expect(!allows(g, TC + "\n<function=run>\n<parameter=command>\necho </tool_call>\n</parameter>\n</function>\n" + TCE), "refuses a call tag's text inside a parameter value");
    expect(!allows(g, TC + "\n<function=run>\n<parameter=command>\necho " + TCE + "\n</parameter>\n</function>\n" + TCE), "refuses a call tag token inside a parameter value");
    expect(!allows(g, "Checking." + TC + "\n<function=run>\n</function>\n" + TCE), "refuses <tool_call> not right after a newline");
    expect(!allows(g, TC + "\n<function=run>\n</function>" + TCE), "refuses </tool_call> not right after a newline");
    expect(!allows(g, kCall + " " + kCall), "refuses a second call not on a new line");
    expect(!accepts(g, TC + "\n<function=run>\n<parameter=command>\nls\n</parameter>\n"), "an open call is not an end of turn");
    expect(allows(g, TC + "\n<function=run>\n<parameter=command>\nls\n</parameter>\n"), "... but may continue");

    const auto required = tool_grammar::compile(tools, "required");
    expect(!allows(required, "Here is the reading list."), "required: text is refused (the answer is the call)");
    expect(!allows(required, "Checking.\n" + kCall), "required: no text before the call either");
    expect(accepts(required, kCall), "required: a call alone");
    expect(accepts(required, kCall + "\n" + kCall), "required: two calls");

    const auto named = tool_grammar::compile(tools, "read_file");
    expect(accepts(named, TC + "\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n</function>\n" + TCE), "named: the named tool");
    expect(!allows(named, kCall), "named: another tool is refused");
    bool threw = false;
    try {
        tool_grammar::compile(tools, "nope");
    } catch (const Error &) {
        threw = true;
    }
    expect(threw, "named: a tool not among the tools is an error");

    // Every call the grammar accepts must parse (serve/output_parser's parse_tool_call), empty values included: a
    // steered model at scale 8 wrote `<parameter=offset>\n</parameter>`.
    {
        const char *bodies[] = {
            "\n<function=read_file>\n<parameter=path>\n/x/README.md\n</parameter>\n<parameter=offset>\n</parameter>\n<parameter=limit>\n120\n</parameter>\n</function>\n",
            "\n<function=run>\n<parameter=command>\n\n</parameter>\n</function>\n",
            "\n<function=run>\n<parameter=command></parameter>\n</function>\n",
            "\n<function=experts>\n</function>\n",
        };
        for (const char *body : bodies) {
            expect(accepts(g, TC + body + TCE), "parser: the grammar accepts the call");
            ToolCallOut call;
            bool parsed = false;
            try {
                parsed = parse_tool_call(body, &tools, call);
            } catch (const std::exception &e) {
                std::printf("      parse_tool_call threw: %s\n", e.what());
            }
            expect(parsed, "parser: parse_tool_call accepts a call the grammar accepts");
        }
    }

    if (argc > 1) {
        // With the tokenizer: <tool_call> and </tool_call> are added tokens, which the mask never allows;
        // StructuredOutput feeds them as the sentinels and lets them through exactly where the grammar accepts those,
        // refuses their text spelling, refuses raw 0xFF tokens, and a whole call feeds through to a complete answer.
        const Tokenizer tok(argv[1]);
        const int32_t call_begin = tok.id_of("<tool_call>"), call_end = tok.id_of("</tool_call>");
        const int32_t think_end = tok.id_of("</think>"), im_end = tok.id_of("<|im_end|>");
        expect(tok.is_added(call_begin) && tok.is_added(call_end), "tokenizer: the call tags are added tokens");
        int32_t raw_ff = -1;  // the plain token whose bytes are the single byte 0xFF (encode refuses invalid UTF-8)
        for (int32_t id = 0; id < tok.size() && raw_ff < 0; ++id)
            if (!tok.is_added(id) && tok.token_bytes(id) == "\xff") raw_ff = id;
        expect(raw_ff >= 0, "tokenizer: a raw 0xFF byte token exists");
        const TokenMasker masker(tok, tok.size());
        auto automaton = std::make_shared<grammar::GrammarAutomaton>(g);
        const std::vector<std::pair<int32_t, std::string>> tags{{call_begin, TC}, {call_end, TCE}};
        StructuredOutput so(automaton, masker, tok, think_end, {im_end}, true, tags);
        for (int32_t t : tok.encode("Checking the shelf.\n")) so.feed(t);
        expect(so.allows(so.cursor(), call_begin), "masks: <tool_call> allowed after a newline");
        expect(so.allows(so.cursor(), im_end), "masks: the end of the turn allowed after text");
        {
            const LogitMasks m = so.masks_for({});
            expect(TokenMasker::allowed(m.bits, call_begin), "masks: the row mask has the <tool_call> bit");
            expect(!TokenMasker::allowed(m.bits, call_end), "masks: the row mask lacks </tool_call> outside a call");
            expect(raw_ff < 0 || !TokenMasker::allowed(m.bits, raw_ff), "masks: the row mask lacks the raw 0xFF byte token");
        }
        StructuredOutput so2(automaton, masker, tok, think_end, {im_end}, true, tags);
        for (int32_t t : tok.encode("Checking.")) so2.feed(t);
        expect(!so2.allows(so2.cursor(), call_begin), "masks: <tool_call> refused when not right after a newline");
        bool fed_ok = true;
        for (int32_t t : tok.encode(kCallText)) {
            if (!so.allows(so.cursor(), t)) {
                fed_ok = false;
                break;
            }
            so.feed(t);
        }
        expect(fed_ok && so.complete(), "masks: a whole call feeds through the tag tokens to a complete answer");
        expect(so.allows(so.cursor(), im_end), "masks: the end of the turn allowed after the call");
        StructuredOutput plain(automaton, masker, tok, think_end, {im_end}, true);
        for (int32_t t : tok.encode("Checking.\n")) plain.feed(t);
        expect(!plain.allows(plain.cursor(), call_begin), "masks: without tag tokens (a response_format grammar) <tool_call> stays refused");
    }

    std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
