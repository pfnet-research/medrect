"""MEDEC Screening task templates for quality assessment."""

from typing import List, Dict
from .base import BaseTemplate
from ..data.medec.samples import MEDECSample


class MEDECScreeningTemplate(BaseTemplate):
    """Template for MEDEC quality screening tasks."""

    def __init__(self, template_name: str):
        """Initialize template with name."""
        super().__init__(template_name)

    # Template definitions
    TEMPLATES = {
        "detailed_explanation_ja": (
            "## JMedecベンチマークの品質評価\n\n"
            "JMedecは、医師国家試験の選択肢を症例文章に変換することで、"
            "LLMのエラー検出・修正能力を評価するベンチマークです。\n\n"
            "### データ生成の仕組み\n"
            "- **CORRECTデータ**: 正解選択肢の内容を症例文に組み込む\n"
            "- **ERRORデータ**: 不正解選択肢の内容を症例文に組み込み、意図的にエラーを作る\n\n"
            "つまり、選択肢の内容が症例文に含まれることは**正常な仕様**です。\n\n"
            "### 元の医師国家試験問題\n"
            "問題文: {original_question}\n"
            "選択肢:\n{choices_text}\n"
            "正解: {correct_answer}\n"
            "**使用選択肢: {used_choice}. {used_choice_text}**\n\n"
            "### 生成されたMEDEC形式テキスト\n"
            "{sentences}\n\n"
            "### エラー情報\n"
            "エラーフラグ: {error_flag}\n"
            "{error_info}\n\n"
            "### 評価タスク\n"
            "上記の生成テキストについて、**ベンチマーク問題としての品質**を評価してください。\n"
            "注意: 選択肢の内容を文章化することは正常です。それ自体をエラーと判定しないでください。\n\n"
            "以下の観点で問題がある場合は1、問題ない場合は0を返してください：\n\n"
            "1. **ambiguous_error**: エラーの判定が曖昧\n"
            "   - 医学的に正しいとも誤りともとれる記述がある\n"
            "   - 複数の解釈が可能で一意に定まらない\n\n"
            "2. **extra_elements**: 元の問題・選択肢にない要素の追加\n"
            "   - 選択肢の文章化は除く（これは正常）\n"
            "   - 問題文や選択肢に存在しない新たな医学情報の追加\n\n"
            "3. **multiple_errors**: ERRORデータで複数のエラー箇所\n"
            "   - 1つの不正解選択肢から複数のエラーが生成されている\n"
            "   - エラー箇所が分散していて特定が困難\n\n"
            "4. **numerical_error**: 数値エラーで修正困難\n"
            "   - ERRORデータで数値の誤りが含まれる\n"
            "   - 文脈から正しい数値を推定するのが困難\n\n"
            "5. **synthesis_consistency_error**: ERRORデータの合成一貫性に問題\n"
            "   - 不正解選択肢を使用しているが医学的に正しい内容になっている\n"
            "   - JMEDECの仕様では不正解選択肢を間違った情報として扱うべき\n\n"
            "JSON形式で回答：\n"
            "{{\n"
            '    "ambiguous_error": 0,\n'
            '    "extra_elements": 0,\n'
            '    "multiple_errors": 0,\n'
            '    "numerical_error": 0,\n'
            '    "synthesis_consistency_error": 0,\n'
            '    "explanation": "判定理由の簡潔な説明"\n'
            "}}"
        ),
        "with_examples_ja": (
            "## JMedecベンチマーク品質評価タスク\n\n"
            "JMedecは医師国家試験の**選択肢を症例文に変換**して作成されるベンチマークです。\n"
            "重要: 選択肢の内容が文章化されることは**設計通り**であり、エラーではありません。\n\n"
            "### 評価例\n\n"
            "#### 例1: CORRECTデータ（正解選択肢を文章化）\n"
            "元の問題: 精巣腫瘍のまず行うべき対応は？ 正解: d. 高位精巣摘除術\n"
            "**使用選択肢: d. 高位精巣摘除術** (正解選択肢)\n"
            "生成文:\n"
            "1. 32歳の男性、右陰囊の腫大を主訴に来院した。\n"
            "（中略）\n"
            "9. これらの所見から、精巣腫瘍が疑われ、まず行うべき対応として高位精巣摘除術を計画した。\n"
            "エラーフラグ: なし\n"
            "→ **正しい**: 正解選択肢を文章化しており、エラーフラグ「なし」は適切\n\n"
            "#### 例2: ERRORデータ（不正解選択肢を正しく文章化）\n"
            "元の問題: 高カリウム血症患者にまず投与すべき薬剤は？ 正解: c. グルコン酸カルシウム\n"
            "**使用選択肢: d. エリスロポエチン製剤** (不正解選択肢)\n"
            "生成文:\n"
            "1. 64歳の女性。息苦しさを主訴に来院した。\n"
            "（中略）\n"
            "13. K 6.7 mEq/L、Ca 7.2 mg/dL、P 5.6 mg/dL。\n"
            "15. 心電図所見：テント状T波を認める。\n"
            "16. 処置：貧血の改善のために、まずエリスロポエチン製剤を投与した。\n"
            "エラーフラグ: あり（文16がエラー）\n"
            "→ **正しい**: 不正解選択肢を使い、緊急性の高い高カリウム血症を無視した間違った治療を文章化\n\n"
            "#### 例3: 合成品質エラー（不正解選択肢を使ったが医学的に正しい内容）\n"
            "元の問題: アセチルコリン負荷冠動脈造影に影響する薬剤は？ 正解: c. カルシウム拮抗薬\n"
            "**使用選択肢: a. スタチン** (不正解選択肢)\n"
            "生成文:\n"
            "1. 65歳女性が胸痛を主訴に来院。高血圧症、脂質異常症、糖尿病、骨粗鬆症の治療歴あり。\n"
            "（中略）\n"
            "7. トロポニンT陰性。アセチルコリン負荷造影を予定。スタチンは検査直前まで継続可と判断。\n"
            "エラーフラグ: あり（文7がエラー）\n"
            "→ **合成品質エラー**: 不正解選択肢「スタチン」を使用しているが、医学的に正しい内容。\n"
            "   JMEDECでは不正解選択肢を**間違った情報として**文章化すべき。\n"
            "   本来は「スタチンは検査結果に影響するため休薬」など医学的に誤った内容にすべき。\n"
            "   synthesis_consistency_error = 1\n\n"
            "#### 例4: 複数エラー（指摘すべき箇所が複数）\n"
            "元の問題: Wilson病でみられる所見は？ 正解: d. Kayser-Fleischer輪\n"
            "**使用選択肢: a. 円錐角膜** (不正解選択肢)\n"
            "生成文:\n"
            "5. コーンメトリー検査で円錐角膜を認め、角膜中央部の進行性弾性張力低下が確認された。\n"
            "10. Cu 25 μg/dL(基準68~128)、セルロプラスミン12 mg/dL(基準21~37)。\n"
            "エラー指摘: 文5のみ（不正解選択肢a「円錐角膜」を使用）\n"
            "→ **問題あり**: 文5では不正解選択肢を使用し、文10では銅とセルロプラスミンの数値が\n"
            "   典型的なWilson病パターン（低値）なのに説明が不十分。複数の観点で問題あり\n"
            "   multiple_errors = 1\n\n"
            "### 今回の評価対象\n\n"
            "#### 元の医師国家試験問題\n"
            "問題文: {original_question}\n"
            "選択肢:\n{choices_text}\n"
            "正解: {correct_answer}\n"
            "**使用選択肢: {used_choice}. {used_choice_text}**\n\n"
            "#### 生成されたMEDEC形式テキスト\n"
            "{sentences}\n\n"
            "#### エラー情報\n"
            "エラーフラグ: {error_flag}\n"
            "{error_info}\n\n"
            "### 評価基準（問題がある=1、問題ない=0）\n\n"
            "1. **ambiguous_error**: エラーの判定が曖昧\n"
            "   - 医学的に正しいとも誤りともとれる記述がある\n"
            "   - 複数の解釈が可能で一意に定まらない\n\n"
            "2. **extra_elements**: 元の問題・選択肢にない要素の追加\n"
            "   - 選択肢の文章化は除く（これは正常）\n"
            "   - 問題文や選択肢に存在しない新たな医学情報の追加\n\n"
            "3. **multiple_errors**: ERRORデータで複数のエラー箇所\n"
            "   - 1つの不正解選択肢から複数のエラーが生成されている\n"
            "   - エラー箇所が分散していて特定が困難\n\n"
            "4. **numerical_error**: 数値エラーで修正困難\n"
            "   - ERRORデータで数値の誤りが含まれる\n"
            "   - 文脈から正しい数値を推定するのが困難\n\n"
            "5. **synthesis_consistency_error**: ERRORデータの合成一貫性に問題\n"
            "   - 不正解選択肢を使用しているが医学的に正しい内容になっている\n"
            "   - JMEDECの仕様では不正解選択肢を間違った情報として扱うべき\n"
            "   - 例: 「スタチンは継続可」→本来は「スタチンは影響するため休薬」とすべき\n\n"
            "JSON形式で回答：\n"
            "{{\n"
            '    "ambiguous_error": 0,\n'
            '    "extra_elements": 0,\n'
            '    "multiple_errors": 0,\n'
            '    "numerical_error": 0,\n'
            '    "synthesis_consistency_error": 0,\n'
            '    "explanation": "判定理由"\n'
            "}}"
        ),
        "detailed_explanation_en": (
            "## MEDEC Benchmark Quality Assessment\n\n"
            "MEDEC (Medical Error Detection and Correction) is a benchmark that evaluates LLMs' ability to detect and correct medical errors in clinical case presentations.\n\n"
            "### Data Structure\n"
            "- **CORRECT data**: Clinical cases without errors (error_flag = 0)\n"
            "- **ERROR data**: Clinical cases with intentional medical errors (error_flag = 1)\n\n"
            "### Clinical Case Information\n"
            "**Case Presentation:**\n{sentences}\n\n"
            "**Error Information:**\n"
            "Error Flag: {error_flag}\n"
            "Error Type: {error_type}\n"
            "{error_info}\n\n"
            "### Assessment Task\n"
            "Please evaluate the **quality of this benchmark sample** from the perspective of medical education and error detection training.\n\n"
            "Rate each criterion as 1 (problematic) or 0 (acceptable):\n\n"
            "1. **ambiguous_error**: Error detection is ambiguous\n"
            "   - The medical statement could be interpreted as either correct or incorrect\n"
            "   - Multiple interpretations are possible, making error detection unclear\n\n"
            "2. **multiple_errors**: Multiple error points in ERROR data\n"
            "   - The case contains multiple distinct medical errors\n"
            "   - Error detection becomes difficult due to scattered error locations\n\n"
            "3. **numerical_error**: Numerical errors that are difficult to correct\n"
            "   - ERROR data contains numerical mistakes\n"
            "   - Correct values cannot be easily inferred from context\n\n"
            "4. **unrealistic_scenario**: Clinically unrealistic patient presentation\n"
            "   - The clinical scenario is medically implausible\n"
            "   - Patient presentation doesn't align with real-world medical practice\n\n"
            "5. **inconsistent_context**: Error doesn't match clinical context\n"
            "   - The identified error is inconsistent with the overall clinical picture\n"
            "   - Error seems forced or unnatural within the case narrative\n\n"
            "Respond in JSON format:\n"
            "{{\n"
            '    "ambiguous_error": 0,\n'
            '    "multiple_errors": 0,\n'
            '    "numerical_error": 0,\n'
            '    "unrealistic_scenario": 0,\n'
            '    "inconsistent_context": 0,\n'
            '    "explanation": "Brief explanation of the assessment"\n'
            "}}"
        ),
        "with_examples_en": (
            "## MEDEC Benchmark Quality Assessment\n\n"
            "MEDEC evaluates LLMs' ability to detect and correct medical errors in clinical cases.\n\n"
            "### Assessment Examples\n\n"
            "#### Example 1: Good ERROR Data (Clear Organism Misidentification)\n"
            "Case: A 7-year-old boy with finger wound, fever, and red streaks on forearm...\n"
            "Error: 'Staphylococcus aureus is the causative agent of the patient's condition.'\n"
            "Correct: 'Group A beta-hemolytic Streptococcus is the causative agent...'\n"
            "Error Type: causalOrganism\n"
            "→ **Good quality**: Clear lymphangitis case, unambiguous organism error\n\n"
            "#### Example 2: Problematic - Ambiguous Error (Diagnostic Uncertainty)\n"
            "Case: 33-year-old woman with end-of-day weakness, diplopia...\n"
            "Error: 'Patient is diagnosed with Guillain-Barre syndrome after physical exam is notable for 2/5 strength of the upper extremities and 4/5 strength of the lower extremities.'\n"
            "Correct: 'Patient is diagnosed with myasthenia gravis...'\n"
            "→ **Problem**: Both conditions can present with weakness; pattern (upper>lower weakness, diplopia, end-of-day fatigue) could suggest either condition depending on interpretation\n"
            "   ambiguous_error = 1\n\n"
            "#### Example 3: Good ERROR Data (Clear Geographic/Travel Medicine Error)\n"
            "Case: 29-year-old returning from Brazil with fever, diarrhea, rash, swimming history...\n"
            "Error: 'Patient's symptoms are suspected to be due to hepatitis A.'\n"
            "Correct: 'Patient's symptoms are suspected to be due to Schistosoma mansoni.'\n"
            "Error Type: causalOrganism\n"
            "→ **Good quality**: Water exposure history makes schistosomiasis obvious; hepatitis A doesn't explain rash\n\n"
            "#### Example 4: Problematic - Inconsistent Context\n"
            "Case: Patient with spinal cord ependymoma and left leg weakness...\n"
            "Error: 'Patient is diagnosed with right-sided Brown-Sequard (hemisection).'\n"
            "Clinical findings: 1/5 left leg strength, decreased vibration/position sensation in left leg, decreased pain/temperature in right leg\n"
            "→ **Problem**: Clinical findings actually support LEFT-sided Brown-Sequard, not right-sided\n"
            "   inconsistent_context = 1\n\n"
            "### Current Assessment Target\n\n"
            "**Case Presentation:**\n{sentences}\n\n"
            "**Error Information:**\n"
            "Error Flag: {error_flag}\n"
            "Error Type: {error_type}\n"
            "{error_info}\n\n"
            "### Assessment Criteria (1=problematic, 0=acceptable)\n\n"
            "1. **ambiguous_error**: Error detection is ambiguous\n"
            "   - Medical statement could be interpreted as correct or incorrect\n"
            "   - Multiple valid interpretations exist\n\n"
            "2. **multiple_errors**: Multiple error points in ERROR data\n"
            "   - Case contains multiple distinct medical errors\n"
            "   - Error locations are scattered and difficult to identify\n\n"
            "3. **numerical_error**: Numerical errors difficult to correct\n"
            "   - Contains numerical mistakes\n"
            "   - Correct values cannot be inferred from context\n\n"
            "4. **unrealistic_scenario**: Clinically unrealistic presentation\n"
            "   - Medical scenario is implausible\n"
            "   - Doesn't align with real-world medical practice\n\n"
            "5. **inconsistent_context**: Error inconsistent with clinical context\n"
            "   - Error doesn't fit the overall clinical picture\n"
            "   - Seems forced or unnatural within the narrative\n\n"
            "Respond in JSON format:\n"
            "{{\n"
            '    "ambiguous_error": 0,\n'
            '    "multiple_errors": 0,\n'
            '    "numerical_error": 0,\n'
            '    "unrealistic_scenario": 0,\n'
            '    "inconsistent_context": 0,\n'
            '    "explanation": "Assessment reasoning"\n'
            "}}"
        ),
    }

    def generate_messages(self, sample: MEDECSample) -> List[Dict[str, str]]:
        """Generate conversation messages for MEDEC screening sample data."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")

        template_string = self.get_template_string()

        # Determine template language and format accordingly
        if self.template_name.endswith("_ja"):
            # Japanese MEDEC data format - with original question information
            metadata = getattr(sample, "metadata", {})
            original_jmle_data = metadata.get("original_jmle_data", {})

            # Format original question and choices
            original_question = original_jmle_data.get("question", "情報なし")
            choices = original_jmle_data.get("choices", {})
            correct_answer = ", ".join(original_jmle_data.get("answer", ["情報なし"]))

            # Format choices text
            choices_text = "\n".join(
                [f"{key}: {value}" for key, value in choices.items()]
            )

            # Extract used choice from sample_id (format: {question_id}_{choice}_{model})
            used_choice = "情報なし"
            used_choice_text = "情報なし"
            try:
                sample_id = getattr(sample, "sample_id", "")
                if sample_id and "_" in sample_id:
                    parts = sample_id.split("_")
                    if len(parts) >= 2:
                        choice_key = parts[
                            1
                        ]  # Extract choice (e.g., 'a', 'b', 'c', etc.)
                        if choice_key in choices:
                            used_choice = choice_key
                            used_choice_text = choices[choice_key]
            except Exception:
                # Fallback to default if parsing fails
                pass

            # Format error information for Japanese templates
            if sample.error_flag == 1:
                error_info = (
                    f"エラー文番号: {sample.error_sentence_id}\n"
                    f"エラー文: {sample.error_sentence}\n"
                    f"修正文: {sample.corrected_sentence}"
                )
            else:
                error_info = "エラーなし"

            # Format template with Japanese MEDEC data
            user_content = self.format_template(
                template_string,
                original_question=original_question,
                choices_text=choices_text,
                correct_answer=correct_answer,
                used_choice=used_choice,
                used_choice_text=used_choice_text,
                sentences=sample.sentences,
                error_flag="あり" if sample.error_flag == 1 else "なし",
                error_info=error_info,
            )

            # System prompt for Japanese templates
            if (
                "detailed_explanation" in self.template_name
                or "with_examples" in self.template_name
            ):
                system_content = (
                    "あなたはJMedecベンチマークの品質管理専門家です。\n"
                    "JMedecは医師国家試験の選択肢を症例文に変換するベンチマークで、"
                    "選択肢の内容が文章化されることは正常な仕様です。\n"
                    "CORRECTデータ（正解選択肢を文章化）とERRORデータ（不正解選択肢を文章化）を"
                    "適切に区別し、エラーフラグの妥当性を評価してください。"
                )
            else:
                system_content = "あなたは医学教育の専門家として、医師国家試験問題を基にしたベンチマーク問題の品質を評価します。医学的正確性と問題設計の妥当性を厳密に判定してください。"

        elif self.template_name.endswith("_en"):
            # English MEDEC data format - no original question information
            error_type_display = sample.error_type if sample.error_type else "None"

            # Format error information for English templates
            if sample.error_flag == 1:
                error_info = (
                    f"Error Sentence ID: {sample.error_sentence_id}\n"
                    f"Error Sentence: {sample.error_sentence}\n"
                    f"Corrected Sentence: {sample.corrected_sentence}"
                )
            else:
                error_info = "No errors"

            # Format template with English MEDEC data
            user_content = self.format_template(
                template_string,
                sentences=sample.sentences,
                error_flag="Yes" if sample.error_flag == 1 else "No",
                error_type=error_type_display,
                error_info=error_info,
            )

            # System prompt for English templates
            system_content = (
                "You are a medical education expert specializing in quality assessment of "
                "medical error detection benchmarks. MEDEC is a benchmark that evaluates "
                "LLMs' ability to detect and correct medical errors in clinical case presentations. "
                "Please evaluate the quality and appropriateness of the provided benchmark sample "
                "from the perspective of medical education and error detection training."
            )

        else:
            raise ValueError(
                f"Template name must end with '_ja' or '_en', got: {self.template_name}"
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        return messages
