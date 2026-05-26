"""
Centralized locale strings for OnePred training pipeline.

All prompts, labels, regex patterns, and format strings are defined here.
Other modules import from this file.
"""

# ============================================================
# Agentic System Prompt
# ============================================================

SYSTEM_PROMPT_TEMPLATE = (
    "You are a user intent prediction model. You receive conversation history turn by turn.\n\n"
    "CRITICAL RULES:\n"
    "1. Intermediate turns: output ONLY <memory>. NEVER output <prediction> until the final turn.\n"
    "2. Final turn (you will see [System Instruction]): output <memory> THEN <prediction> with exactly 3 candidate questions.\n"
    "3. Each candidate must be a single question. No multi-part questions.\n"
    "4. Keep your thinking brief and focused. Do NOT over-analyze or repeat yourself.\n"
    "5. Always respond in English.\n\n"
    "Memory guidelines:\n"
    "- Record what helps predict the next question: topic shifts, user's unresolved needs, conversation trajectory.\n"
    "- Keep memory concise (under 1500 tokens). Merge and compress old info, never dump raw conversation.\n"
    "- Update each turn: keep relevant old info, add new observations.\n\n"
    "Output format:\n"
    "- Intermediate: <memory>key observations for predicting next question</memory>\n"
    "- Final: <memory>key observations for predicting next question</memory><prediction>\n1. candidate question 1\n2. candidate question 2\n3. candidate question 3\n</prediction>"
)

# ============================================================
# Fullhistory System Prompt
# ============================================================

FULLHISTORY_SYSTEM_PROMPT_TEMPLATE = (
    "You are a user intent prediction model.\n\n"
    "Current user profile (for reference only, use briefly, do not analyze item by item):\n{user_profile}\n\n"
    "You will see a conversation history. Based on the conversation content, predict the most likely next question the user will ask.\n\n"
    "Each candidate must be a single question. It is strictly forbidden to include multiple questions or follow-ups in one candidate.\n\n"
    "Please provide 3 candidate questions in <prediction>.\n"
    "Output format:\n"
    "<prediction>\n"
    "1. Candidate question 1\n"
    "2. Candidate question 2\n"
    "3. Candidate question 3\n"
    "</prediction>"
)

# ============================================================
# Memory User Template (recurrent agent loop)
# ============================================================

MEMORY_USER_TEMPLATE = (
    "Your previous memory:\n"
    "<memory>{memory}</memory>\n\n"
    "{observation}\n\n"
    "Update your memory in <memory> tags. Keep it concise and specific to the conversation. "
    "Think briefly, then output the tag immediately."
)

# ============================================================
# Observation format
# ============================================================


def observation_header(turn_idx: int) -> str:
    return f"[Turn {turn_idx}]"


observation_user_label = "User: "
observation_response_label = "Response: "
feedback_label = "User feedback: "

system_instruction = (
    "\n\n[System Instruction] This is the FINAL turn. You MUST now output BOTH tags:\n"
    "1. <memory>your updated memory</memory>\n"
    "2. <prediction>\n1. candidate question 1\n2. candidate question 2\n3. candidate question 3\n</prediction>\n"
    "Keep thinking brief. Output the tags immediately."
)

# ============================================================
# User profile labels
# ============================================================

profile_long_term_label = "Long-term memory:"
profile_short_term_label = "Short-term memory:"

# ============================================================
# User feedback labels
# ============================================================

feedback_map = {
    "is_like": "liked",
    "is_dislike": "disliked",
    "copy_clk_cnt": "copied",
    "share_clk_cnt": "shared",
    "delete_clk_cnt": "deleted",
    "refresh_clk_cnt": "refreshed",
}
feedback_prefix = "User actions: "
feedback_joiner = ", "

# ============================================================
# Fallbacks & markers
# ============================================================

fallback_none = "None"
fallback_empty = "(empty)"


def build_system_prompt(user_profile: str = "") -> str:
    """Build system prompt with user profile filled in."""
    profile = user_profile.strip() if user_profile else fallback_none
    return SYSTEM_PROMPT_TEMPLATE.format(user_profile=profile)


truncation_marker = "…(truncated)"
response_truncation_marker = "…(truncated)"
brief_reason = "brief reason"

# ============================================================
# Fullhistory format strings
# ============================================================

fullhistory_conversation_header = "Below is the complete conversation history:"
fullhistory_prediction_instruction = "Please predict the most likely next question the user will ask."

# ============================================================
# Prediction block label (listwise judge)
# ============================================================


def prediction_block_label(i: int) -> str:
    return f"Prediction {i}"


# ============================================================
# Pointwise Judge Prompts
# ============================================================

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, stable, low-variance question prediction evaluation expert.\n\n"
    "Your task is: based on the given conversation memory, list of previous user questions, "
    "model-generated candidate next questions, and the user's actual next question (Ground Truth), "
    "independently evaluate the match between each candidate question and the actual next question.\n\n"
    "Please follow these principles:\n\n"
    "1. Your core task is to judge \"whether the candidate question matches the actual next question\", "
    "not \"whether this question looks reasonable\".\n"
    "2. The three candidates must be evaluated independently; do not compare them with each other.\n"
    "3. Do not give bonus points for more complete, polite, or assistant-like wording.\n"
    "4. Do not infer information not provided in the memory; judge only based on the input content.\n"
    "5. Output must strictly follow the specified JSON format; do not output any additional explanation."
)

JUDGE_USER_PROMPT = (
    "Based on the given memory, list of previous user questions, and actual next question, "
    "evaluate the 3 candidate questions separately.\n\n"
    "[Conversation Memory]\n{memory}\n\n"
    "[Previous User Questions]\n{previous_user_queries}\n\n"
    "[Actual Next Question (Ground Truth)]\n{ground_truth}\n\n"
    "[Candidate Question 1]\n{candidate_1}\n\n"
    "[Candidate Question 2]\n{candidate_2}\n\n"
    "[Candidate Question 3]\n{candidate_3}\n\n"
    "Score each candidate question using the following criteria. "
    "Scores must be one of: 0.0, 0.25, 0.5, 0.75, 1.0.\n\n"
    "I. Low-score conditions (any one of these results in 0.0):\n"
    "- The candidate is a meaningless greeting, e.g. \"hello\", \"thanks\", \"goodbye\"\n"
    "- The candidate simply repeats content from [Previous User Questions] rather than predicting a new question\n"
    "- The candidate is too generic, e.g. \"any other questions?\", \"what else would you like to know?\", \"go on\"\n\n"
    "II. Scoring criteria:\n"
    "- 1.0: The candidate and actual next question are semantically identical, same core intent, just different wording\n"
    "- 0.75: The candidate and actual next question are highly related, different wording but essentially the same intent\n"
    "- 0.5: The candidate and actual next question are partially related, shared topic but different focus\n"
    "- 0.25: The candidate and actual next question are slightly related, related topic but clearly different direction\n"
    "- 0.0: The candidate and actual next question are completely unrelated, or meet the low-score conditions above\n\n"
    "Output strictly in JSON format as follows, with no additional text:\n\n"
    "{{\n"
    '  "candidate_1": {{"score": 0.0, "reason": "brief reason"}},\n'
    '  "candidate_2": {{"score": 0.0, "reason": "brief reason"}},\n'
    '  "candidate_3": {{"score": 0.0, "reason": "brief reason"}}\n'
    "}}"
)

# ============================================================
# Listwise Judge Prompts
# ============================================================

LISTWISE_SYSTEM_PROMPT = (
    "You are a strict prediction quality comparison expert.\n\n"
    "Your task is: based on the given list of previous user questions, multiple sets of model-generated "
    "candidate next questions, and the user's actual next question (Ground Truth), compare and determine "
    "which set of predictions best matches the user intent of the actual next question, and provide a ranking.\n\n"
    "Please follow these principles:\n"
    "1. The core criterion is \"whether the candidate questions match the intent of the actual next question\", "
    "not \"whether the question looks good\".\n"
    "2. Each prediction set contains up to 3 candidate questions; as long as any one candidate matches "
    "the actual next question, that set should receive a better ranking.\n"
    "3. Do not give better rankings for more complete, polite, or assistant-like wording.\n"
    "4. Judge only based on the input content; do not infer information not provided.\n"
    "5. Empty or meaningless predictions should rank last.\n"
    "6. Output must strictly follow the specified JSON format; do not output any additional explanation."
)

LISTWISE_USER_PROMPT = (
    "Compare the following {n} sets of predictions and determine which set's candidate questions "
    "best match the user intent of the actual next question, then provide a ranking.\n\n"
    "[Previous User Questions]\n{previous_queries}\n\n"
    "[Actual Next Question (Ground Truth)]\n{ground_truth}\n\n"
    "{prediction_blocks}\n\n"
    "Please use the following matching criteria to rank each prediction set:\n\n"
    "I. Low-score conditions (any one of these should rank last):\n"
    "- The candidate is a meaningless greeting, e.g. \"hello\", \"thanks\", \"goodbye\"\n"
    "- The candidate simply repeats content from [Previous User Questions] rather than predicting a new question\n"
    "- The candidate is too generic, e.g. \"any other questions?\", \"what else would you like to know?\", \"go on\"\n\n"
    "II. Matching levels (from high to low):\n"
    "- Intent match: candidate and actual next question are semantically identical, same core intent, just different wording\n"
    "- Direction correct: candidate and actual next question are highly related, different wording but essentially the same intent\n"
    "- Topic related: candidate and actual next question are partially related, shared topic but different focus\n"
    "- Reasonable question: candidate and actual next question are slightly related, related topic but clearly different direction\n"
    "- Completely unrelated: candidate and actual next question are completely unrelated, or meet the low-score conditions above\n\n"
    "Please rank these {n} prediction sets by matching level.\n"
    "- Ranking from 1 (best match) to {n} (worst match)\n"
    "- Tied rankings are allowed (same ranking when matching levels are equal)\n"
    "- For each prediction set, use its best candidate's matching level as the basis for ranking\n\n"
    "Output strictly in JSON format as follows, with no additional text:\n\n"
    "{json_template}"
)

# ============================================================
# Regex Filters
# ============================================================

is_not_question_patterns = [
    r"(?i)^the user (might|may|will|would|could|probably|is likely to)",
    r"(?i)^(he|she|the customer|the client) (might|may|will|would|could)",
    r"(?i)(the user might|the user wants to know|the user will next)",
    r"(?i)^inquire\b",
    r"(?i)^request (to provide|for|that)",
    r"(?i)^(learn|know) more about",
]

is_generic_patterns = [
    r"(?i)^(anything else|continue|more|go on)\s*[?]?\s*$",
    r"(?i)^can you (explain|describe|tell me more)\s*[?]?\s*$",
    r"(?i)^(are there|do you have) (any )?(other|more) (options|suggestions|recommendations|choices)\s*[?]?\s*$",
    r"(?i)^how (exactly|specifically) (do|should|can) (I|we|you) do (it|this|that)\s*[?]?\s*$",
    r"(?i)^can you provide more (information|details)\s*[?]?\s*$",
    r"(?i)^are there (similar|alternative) (options|choices|recommendations)\s*[?]?\s*$",
]

# ============================================================
# Observation parsing patterns (bilingual for backward compat with existing data)
# ============================================================

parse_query_pattern = r"(?:User):\s*(.*?)\n(?:Response):"
parse_response_pattern = r"(?:Response):\s*(.+?)(?:\n(?:User feedback):|\n\n\[(?:System Instruction)\]|$)"
parse_feedback_pattern = r"(?:User feedback):\s*(.+?)(?:\n\n\[(?:System Instruction)\]|$)"
parse_turn_idx_pattern = r"\[Turn (\d+)\]"
