#pragma once

// Tool calls -> byte grammar (serve/grammar): the shape of an answer that may call the request's tools, so that a
// generation under activation steering (or any pressure that bends syntax) cannot emit a call the parser would
// read as text. Free text is unconstrained except that it never contains a `<tool_call>`, `</tool_call>` or `<function=` tag; once a call opens
// it must be the `<tool_call>` token, `<function=NAME>` with NAME one of the request's tools, zero or more
// `<parameter=P>` ... `</parameter>` with P one of that tool's `parameters.properties` (any name when the tool
// declares no properties), `</function>`, a newline, the `</tool_call>` token. Parameter values are any text that does not
// contain `</parameter>` or a call tag; whether a required parameter is present is not enforced (the OutputParser still checks
// the call against the schema). White space between tags is bounded (kMaxWhitespaceRun), and a `<tool_call>` or
// `</tool_call>` stands right after a newline, as the parser's literal-tags rule requires.
//
// choice: "auto" - calls optional (text alone, or text ending in a newline then one or more calls);
//         "required" - the answer is one or more calls, no text before them; a tool's name - the same, that tool only.
// Throws strix::Error for a tool or property name the format cannot carry ('>' or a newline in it), or a named
// choice that is not among the tools.

#include "serve/grammar.hpp"
#include "serve/json.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace strix::tool_grammar {

constexpr int kMaxWhitespaceRun = 16;

// The call tags in the grammar are these byte sentinels, not the tags' text: the parser recognizes the tokenizer's
// added tokens <tool_call> and </tool_call>, and a model under pressure can spell the same text with ordinary tokens,
// which the parser then reads as text (seen at steering scale 8). StructuredOutput feeds a tag token as its sentinel,
// so only the real token opens or closes a call; 0xFF never occurs in UTF-8 text, and the raw 0xFF byte tokens are
// refused under this grammar, so no plain token can spell a sentinel.
constexpr std::string_view kCallBegin = "\xff\x01", kCallEnd = "\xff\x02";

std::shared_ptr<const grammar::Grammar> compile(const json::Value &tools, const std::string &choice);

}  // namespace strix::tool_grammar
