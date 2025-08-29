"""
OASIS Rule Engine Library
A rule-based engine for validating and modifying OASIS M-Codes and GG-Codes answers
based on patient walker usage and other clinical conditions.
Supports natural language questions mapped to appropriate codes.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
from difflib import get_close_matches


class WalkerStatus(Enum):
    """Walker usage status for the patient"""
    NO_WALKER = "no_walker"
    USES_WALKER = "uses_walker"
    WHEELCHAIR_SOME = "wheelchair_some"
    WHEELCHAIR_ONLY = "wheelchair_only"


class CodeType(Enum):
    """Type of OASIS code"""
    M_CODE = "M"
    GG_CODE = "GG"


@dataclass
class RuleResult:
    """Result of rule application"""
    original_answer: Union[str, int]
    modified_answer: Union[str, int]
    rule_applied: str
    confidence: float
    warnings: List[str]
    detected_code: Optional[str] = None
    detected_question: Optional[str] = None


class OASISRuleEngine:
    """Rule engine for OASIS M-Codes and GG-Codes validation and modification"""
    
    def __init__(self):
        self._initialize_rules()
        self._initialize_question_mappings()
        
    def _initialize_rules(self):
        """Initialize the rule definitions based on the document"""
        
        # M-Code rules
        self.m_code_rules = {
            "M1860": {  # Ambulation/Locomotion
                "walker_scores": {
                    "single_point_cane": 2,
                    "crutch": 2,
                    "walker": 2,
                    "wheelchair_some": 3
                },
                "description": "Ambulation/Locomotion"
            },
            "M1400": {  # Dyspnea
                "conditions": {
                    "moderate_exertion": 2,  # walking across room with walker
                    "minimal_exertion": 3    # walking few steps with walker
                },
                "description": "Dyspnea"
            },
            "M1242": {  # Frequency of Pain
                "walker_related": {
                    "less_than_daily": 2,
                    "daily_not_constant": 3
                },
                "description": "Frequency of Pain Interfering with Activity"
            },
            "M1870": {  # Feeding or Grooming
                "walker_impact": 1,  # walker doesn't directly affect
                "description": "Feeding or Grooming"
            },
            "M1880": {  # Meal Preparation
                "walker_typical": 2,  # Unable to prepare, can heat/serve
                "description": "Ability to Prepare Light Meals"
            }
        }
        
        # GG-Code rules
        self.gg_code_rules = {
            "GG0170C": {  # Lying to Sitting
                "walker_score": 4,  # Supervision or Touching Assist
                "description": "Lying to Sitting on Side of Bed"
            },
            "GG0170D": {  # Sit to Stand
                "walker_score": 4,
                "description": "Sit to Stand"
            },
            "GG0170E": {  # Chair/Bed Transfer
                "walker_score": 4,
                "description": "Chair/Bed-to-Chair Transfer"
            },
            "GG0170F": {  # Toilet Transfer
                "walker_score": 4,
                "description": "Toilet Transfer"
            },
            "GG0170J": {  # Walk 50 Feet
                "walker_score": 4,
                "description": "Walk 50 Feet with Two Turns"
            },
            "GG0170K": {  # Walk 150 Feet
                "walker_able": 4,
                "walker_unsafe": 88,  # Not attempted due to safety
                "description": "Walk 150 Feet"
            },
            "GG0170L": {  # Uneven Surfaces
                "walker_typical": 88,  # Not attempted
                "description": "Walking on Uneven Surfaces"
            },
            "GG0170M": {  # 1 Step (Curb)
                "walker_typical": 88,
                "description": "1 Step (Curb)"
            },
            "GG0170N": {  # 4 Steps
                "walker_typical": 88,
                "description": "4 Steps"
            },
            "GG0170O": {  # 12 Steps
                "walker_typical": 88,
                "description": "12 Steps"
            },
            "GG0170R": {  # Wheel 50 Feet
                "no_wheelchair": 88,
                "description": "Wheel 50 Feet with Two Turns"
            },
            "GG0170S": {  # Wheel 150 Feet
                "no_wheelchair": 88,
                "description": "Wheel 150 Feet"
            }
        }
        
        # Special GG-Code values
        self.gg_special_codes = {
            6: "Independent",
            4: "Supervision or Touching Assist",
            7: "Patient refused",
            9: "Not applicable",
            10: "Not attempted due to environmental limitations",
            88: "Not attempted due to medical condition or safety concerns"
        }
    
    def _initialize_question_mappings(self):
        """Initialize natural language question mappings to codes"""
        
        self.question_mappings = {
            # M-Codes mappings
            "M1860": [
                "ambulation",
                "locomotion",
                "walking ability",
                "how does the patient walk",
                "patient mobility",
                "ability to walk",
                "walking assistance needed",
                "ambulatory status",
                "can patient walk independently"
            ],
            "M1400": [
                "dyspnea",
                "shortness of breath",
                "breathing difficulty",
                "breathlessness",
                "difficulty breathing",
                "respiratory distress",
                "sob",
                "breathing with exertion",
                "breathing problems"
            ],
            "M1242": [
                "pain frequency",
                "how often pain interferes",
                "frequency of pain",
                "pain interfering with activity",
                "pain interference",
                "how often does pain affect activities",
                "pain affecting daily activities"
            ],
            "M1870": [
                "feeding",
                "grooming",
                "eating ability",
                "self feeding",
                "personal hygiene",
                "ability to feed self",
                "grooming ability",
                "feeding or grooming"
            ],
            "M1880": [
                "meal preparation",
                "prepare meals",
                "cooking ability",
                "light meal preparation",
                "ability to make meals",
                "can patient cook",
                "preparing food"
            ],
            
            # GG-Codes mappings
            "GG0170C": [
                "lying to sitting",
                "bed to sitting",
                "sit up in bed",
                "lying to sitting on side of bed",
                "getting up from lying down",
                "sitting up from lying position"
            ],
            "GG0170D": [
                "sit to stand",
                "standing from sitting",
                "chair to standing",
                "getting up from chair",
                "standing up",
                "rising from seated position"
            ],
            "GG0170E": [
                "chair transfer",
                "bed to chair",
                "chair to bed",
                "bed chair transfer",
                "transferring between bed and chair",
                "moving from bed to chair"
            ],
            "GG0170F": [
                "toilet transfer",
                "toilet mobility",
                "getting on toilet",
                "getting off toilet",
                "bathroom transfer",
                "toilet sitting and standing"
            ],
            "GG0170J": [
                "walk 50 feet",
                "walking 50 feet with turns",
                "short distance walking",
                "walk fifty feet",
                "ambulate 50 feet",
                "walking with two turns"
            ],
            "GG0170K": [
                "walk 150 feet",
                "walking 150 feet",
                "longer distance walking",
                "walk one hundred fifty feet",
                "ambulate 150 feet",
                "walking longer distance"
            ],
            "GG0170L": [
                "uneven surfaces",
                "walking on uneven ground",
                "rough terrain walking",
                "uneven surface ambulation",
                "walking on irregular surfaces",
                "ambulating uneven surfaces"
            ],
            "GG0170M": [
                "one step",
                "single step",
                "curb",
                "step up",
                "step down",
                "managing one step",
                "curb navigation"
            ],
            "GG0170N": [
                "four steps",
                "4 steps",
                "few stairs",
                "climbing four steps",
                "managing four steps",
                "short staircase"
            ],
            "GG0170O": [
                "twelve steps",
                "12 steps",
                "full flight of stairs",
                "staircase",
                "climbing twelve steps",
                "managing twelve steps",
                "full stairs"
            ],
            "GG0170R": [
                "wheel 50 feet",
                "wheelchair 50 feet",
                "wheeling short distance",
                "wheelchair mobility 50",
                "propel wheelchair 50 feet"
            ],
            "GG0170S": [
                "wheel 150 feet",
                "wheelchair 150 feet", 
                "wheeling longer distance",
                "wheelchair mobility 150",
                "propel wheelchair 150 feet"
            ]
        }
        
        # Create reverse mapping for quick lookup
        self.natural_to_code = {}
        for code, phrases in self.question_mappings.items():
            for phrase in phrases:
                self.natural_to_code[phrase.lower()] = code
    
    def detect_code_from_question(self, question: str) -> Tuple[Optional[str], float]:
        """
        Detect the appropriate code from a natural language question
        
        Returns:
            Tuple of (detected_code, confidence_score)
        """
        question_lower = question.lower().strip()
        
        # First, check if it's already a code
        if question_lower.startswith('m') or question_lower.startswith('gg'):
            code_upper = question.upper()
            if code_upper in self.question_mappings:
                return code_upper, 1.0
        
        # Direct match
        if question_lower in self.natural_to_code:
            return self.natural_to_code[question_lower], 1.0
        
        # Partial matching - check if key phrases are in the question
        best_match_score = 0
        best_match_code = None
        
        for code, phrases in self.question_mappings.items():
            for phrase in phrases:
                # Calculate similarity score
                phrase_words = set(phrase.lower().split())
                question_words = set(question_lower.split())
                
                # Check for word overlap
                common_words = phrase_words.intersection(question_words)
                if common_words:
                    score = len(common_words) / max(len(phrase_words), len(question_words))
                    
                    # Boost score if phrase is substring
                    if phrase.lower() in question_lower:
                        score += 0.5
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_match_code = code
        
        # Use fuzzy matching as fallback
        if best_match_score < 0.3:
            all_phrases = []
            for phrases in self.question_mappings.values():
                all_phrases.extend(phrases)
            
            matches = get_close_matches(question_lower, all_phrases, n=1, cutoff=0.6)
            if matches:
                best_match_code = self.natural_to_code[matches[0].lower()]
                best_match_score = 0.7
        
        if best_match_code:
            return best_match_code, min(best_match_score, 1.0)
        
        return None, 0.0
    
    def parse_code(self, code: str) -> Tuple[CodeType, str]:
        """Parse the code to determine its type and normalize it"""
        code = code.upper().strip()
        
        if code.startswith("M"):
            return CodeType.M_CODE, code
        elif code.startswith("GG"):
            return CodeType.GG_CODE, code
        else:
            raise ValueError(f"Unknown code type: {code}")
    
    def detect_walker_status(self, context: Dict) -> WalkerStatus:
        """Detect walker status from context"""
        context_str = str(context).lower()
        
        if "wheelchair only" in context_str or "wheelchair bound" in context_str:
            return WalkerStatus.WHEELCHAIR_ONLY
        elif "wheelchair" in context_str and ("some" in context_str or "sometimes" in context_str):
            return WalkerStatus.WHEELCHAIR_SOME
        elif "walker" in context_str or "uses walker" in context_str:
            return WalkerStatus.USES_WALKER
        else:
            return WalkerStatus.NO_WALKER
    
    def apply_m_code_rules(self, code: str, answer: Union[str, int], 
                           walker_status: WalkerStatus, context: Dict) -> RuleResult:
        """Apply M-Code specific rules"""
        warnings = []
        rule_applied = "No rule applied"
        modified_answer = answer
        confidence = 1.0
        
        if code not in self.m_code_rules:
            warnings.append(f"Unknown M-Code: {code}")
            return RuleResult(answer, answer, rule_applied, 0.5, warnings, code, None)
        
        rule = self.m_code_rules[code]
        
        # M1860 - Ambulation/Locomotion
        if code == "M1860":
            if walker_status == WalkerStatus.USES_WALKER:
                modified_answer = 2
                rule_applied = "Walker usage requires score 2 for indoor mobility"
            elif walker_status == WalkerStatus.WHEELCHAIR_SOME:
                modified_answer = 3
                rule_applied = "Wheelchair for some mobility requires score 3"
        
        # M1400 - Dyspnea
        elif code == "M1400":
            context_str = str(context).lower()
            if "moderate exertion" in context_str or "walking across room" in context_str:
                modified_answer = 2
                rule_applied = "Dyspnea with moderate exertion scores 2"
            elif "minimal exertion" in context_str or "few steps" in context_str:
                modified_answer = 3
                rule_applied = "Dyspnea with minimal exertion scores 3"
        
        # M1242 - Pain Frequency
        elif code == "M1242":
            if walker_status == WalkerStatus.USES_WALKER:
                context_str = str(context).lower()
                if "daily" in context_str and "not constant" in context_str:
                    modified_answer = 3
                    rule_applied = "Daily but not constant pain scores 3"
                elif "less" in context_str and "daily" in context_str:
                    modified_answer = 2
                    rule_applied = "Less than daily pain scores 2"
        
        # M1870 - Feeding/Grooming
        elif code == "M1870":
            if walker_status == WalkerStatus.USES_WALKER:
                modified_answer = 1
                rule_applied = "Walker doesn't directly affect feeding/grooming - score 1"
        
        # M1880 - Meal Preparation
        elif code == "M1880":
            if walker_status == WalkerStatus.USES_WALKER:
                modified_answer = 2
                rule_applied = "With walker, typically unable to prepare but can heat/serve - score 2"
        
        return RuleResult(answer, modified_answer, rule_applied, confidence, warnings, code, rule['description'])
    
    def apply_gg_code_rules(self, code: str, answer: Union[str, int], 
                            walker_status: WalkerStatus, context: Dict) -> RuleResult:
        """Apply GG-Code specific rules"""
        warnings = []
        rule_applied = "No rule applied"
        modified_answer = answer
        confidence = 1.0
        
        if code not in self.gg_code_rules:
            warnings.append(f"Unknown GG-Code: {code}")
            return RuleResult(answer, answer, rule_applied, 0.5, warnings, code, None)
        
        rule = self.gg_code_rules[code]
        
        # Key rule: Walker = never score "06 - Independent" for ambulation tasks
        if walker_status == WalkerStatus.USES_WALKER and int(answer) == 6:
            if code in ["GG0170J", "GG0170K", "GG0170L", "GG0170M", "GG0170N", "GG0170O"]:
                warnings.append("Walker users cannot score 06 (Independent) for ambulation tasks")
                modified_answer = 4  # Default to supervision/touching assist
                rule_applied = "Walker users require at least supervision - changed from 06 to 04"
        
        # Basic transfer tasks with walker
        if code in ["GG0170C", "GG0170D", "GG0170E", "GG0170F"]:
            if walker_status == WalkerStatus.USES_WALKER:
                modified_answer = 4
                rule_applied = f"Walker user typically scores 04 (Supervision/Touching) for {rule['description']}"
        
        # Walking tasks
        elif code == "GG0170J":  # Walk 50 feet
            if walker_status == WalkerStatus.USES_WALKER:
                modified_answer = 4
                rule_applied = "Walker user scores 04 for Walk 50 Feet with Two Turns"
        
        elif code == "GG0170K":  # Walk 150 feet
            if walker_status == WalkerStatus.USES_WALKER:
                context_str = str(context).lower()
                if "unsafe" in context_str or "limited" in context_str:
                    modified_answer = 88
                    rule_applied = "Walker user with safety concerns scores 88 for Walk 150 Feet"
                else:
                    modified_answer = 4
                    rule_applied = "Walker user able to walk 150 feet scores 04"
        
        # Challenging surfaces/stairs - typically not attempted with walker
        elif code in ["GG0170L", "GG0170M", "GG0170N", "GG0170O"]:
            if walker_status == WalkerStatus.USES_WALKER:
                modified_answer = 88
                rule_applied = f"Walker makes {rule['description']} unsafe - score 88"
        
        # Wheelchair tasks
        elif code in ["GG0170R", "GG0170S"]:
            if walker_status in [WalkerStatus.NO_WALKER, WalkerStatus.USES_WALKER]:
                modified_answer = 88
                rule_applied = "Patient does not use wheelchair - score 88"
        
        # Check for M1860 and GG0170J/K alignment
        if code in ["GG0170J", "GG0170K"]:
            warnings.append("Ensure alignment with M1860 Ambulation/Locomotion score")
        
        return RuleResult(answer, modified_answer, rule_applied, confidence, warnings, code, rule['description'])
    
    def validate_and_modify(self, question: str, answer: Union[str, int], 
                           context: Optional[Dict] = None) -> RuleResult:
        """
        Main method to validate and modify an answer based on rules
        
        Args:
            question: Natural language question or code (e.g., "How does patient walk?", "M1860", "GG0170J")
            answer: The current answer value
            context: Additional context about the patient (walker usage, conditions, etc.)
        
        Returns:
            RuleResult with original answer, modified answer, rule applied, and warnings
        """
        if context is None:
            context = {}
        
        try:
            # Try to detect the code from natural language
            detected_code, detection_confidence = self.detect_code_from_question(question)
            
            if not detected_code:
                return RuleResult(
                    original_answer=answer,
                    modified_answer=answer,
                    rule_applied="Could not identify OASIS code from question",
                    confidence=0.0,
                    warnings=[f"Unable to match question '{question}' to any OASIS code"],
                    detected_code=None,
                    detected_question=question
                )
            
            # Parse the detected code
            code_type, code = self.parse_code(detected_code)
            
            # Detect walker status from context
            walker_status = self.detect_walker_status(context)
            
            # Apply appropriate rules based on code type
            if code_type == CodeType.M_CODE:
                result = self.apply_m_code_rules(code, answer, walker_status, context)
            else:  # GG_CODE
                result = self.apply_gg_code_rules(code, answer, walker_status, context)
            
            # Update confidence based on detection confidence
            result.confidence = min(result.confidence, detection_confidence)
            
            # Add detection info to warnings if confidence is low
            if detection_confidence < 0.8:
                result.warnings.append(f"Question matched to {code} with {detection_confidence:.0%} confidence")
            
            return result
            
        except Exception as e:
            return RuleResult(
                original_answer=answer,
                modified_answer=answer,
                rule_applied=f"Error: {str(e)}",
                confidence=0.0,
                warnings=[str(e)],
                detected_code=None,
                detected_question=question
            )
    
    def batch_validate(self, qa_pairs: List[Dict]) -> List[RuleResult]:
        """
        Validate multiple question-answer pairs
        
        Args:
            qa_pairs: List of dictionaries with 'question', 'answer', and optional 'context'
        
        Returns:
            List of RuleResults
        """
        results = []
        for pair in qa_pairs:
            result = self.validate_and_modify(
                question=pair.get('question'),
                answer=pair.get('answer'),
                context=pair.get('context', {})
            )
            results.append(result)
        return results
    
    def get_code_info(self, question: str) -> Dict:
        """Get information about a specific code or question"""
        try:
            # Try to detect code from question
            detected_code, confidence = self.detect_code_from_question(question)
            
            if not detected_code:
                return {"error": f"Could not identify code from question: {question}"}
            
            code_type, normalized_code = self.parse_code(detected_code)
            
            info = {
                "detected_code": normalized_code,
                "detection_confidence": f"{confidence:.0%}",
                "original_question": question
            }
            
            if code_type == CodeType.M_CODE:
                if normalized_code in self.m_code_rules:
                    info.update({
                        "type": "M-Code",
                        "details": self.m_code_rules[normalized_code]
                    })
            else:
                if normalized_code in self.gg_code_rules:
                    info.update({
                        "type": "GG-Code",
                        "details": self.gg_code_rules[normalized_code],
                        "special_codes": self.gg_special_codes
                    })
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_supported_questions(self) -> Dict[str, List[str]]:
        """List all supported natural language questions grouped by code"""
        result = {}
        for code, phrases in self.question_mappings.items():
            if code.startswith('M'):
                description = self.m_code_rules.get(code, {}).get('description', '')
            else:
                description = self.gg_code_rules.get(code, {}).get('description', '')
            
            result[f"{code} - {description}"] = phrases
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize the rule engine
    engine = OASISRuleEngine()
    
    # Test cases with natural language questions
    test_cases = [
        {
            "question": "How does the patient walk?",
            "answer": 1,
            "context": {"patient_uses": "walker", "mobility": "indoor"}
        },
        {
            "question": "Can the patient walk 50 feet with turns?",
            "answer": 6,  # Independent - should be modified
            "context": {"patient_uses": "walker"}
        },
        {
            "question": "Walking on uneven surfaces",
            "answer": 4,
            "context": {"patient_uses": "walker", "terrain": "uneven surfaces"}
        },
        {
            "question": "Does patient have shortness of breath?",
            "answer": 1,
            "context": {"dyspnea": "with moderate exertion", "walking": "across room with walker"}
        },
        {
            "question": "Patient ability to walk 150 feet",
            "answer": 6,
            "context": {"patient_uses": "walker", "walking": "limited and unsafe"}
        },
        {
            "question": "Can patient prepare meals?",
            "answer": 1,
            "context": {"patient_uses": "walker"}
        },
        {
            "question": "Toilet transfer ability",
            "answer": 6,
            "context": {"patient_uses": "walker"}
        },
        {
            "question": "M1860",  # Still supports code directly
            "answer": 4,
            "context": {"patient_uses": "wheelchair", "sometimes": "yes"}
        }
    ]
    
    print("OASIS Rule Engine Test Results (Natural Language)")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        result = engine.validate_and_modify(
            question=test["question"],
            answer=test["answer"],
            context=test["context"]
        )
        
        print(f"\nTest Case {i}:")
        print(f"  Question: {test['question']}")
        if result.detected_code:
            print(f"  Detected Code: {result.detected_code} ({result.detected_question})")
        print(f"  Original Answer: {result.original_answer}")
        print(f"  Modified Answer: {result.modified_answer}")
        print(f"  Rule Applied: {result.rule_applied}")
        print(f"  Confidence: {result.confidence:.0%}")
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")
    
    print("\n" + "=" * 60)
    print("Batch Validation Example")
    print("=" * 60)
    
    batch_results = engine.batch_validate(test_cases)
    for i, result in enumerate(batch_results, 1):
        if result.original_answer != result.modified_answer:
            print(f"Case {i}: Changed from {result.original_answer} to {result.modified_answer}")
    
    print("\n" + "=" * 60)
    print("Sample Supported Questions")
    print("=" * 60)
    
    # Show a few examples of supported questions
    supported = engine.list_supported_questions()
    sample_codes = ["M1860 - Ambulation/Locomotion", "GG0170J - Walk 50 Feet with Two Turns"]
    for code_desc in sample_codes:
        if code_desc in supported:
            print(f"\n{code_desc}:")
            for q in supported[code_desc][:3]:  # Show first 3 examples
                print(f"  - {q}")