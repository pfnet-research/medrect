"""MEDEC Error Type Reassignment templates."""

from typing import List, Dict
from .base import BaseTemplate
from ..data.medec.samples import MEDECSample


class MEDECErrorTypeTemplate(BaseTemplate):
    """Template for MEDEC error type reassignment tasks."""

    # Template definitions
    TEMPLATES = {
        "error_type_classification_ja": (
            "## 医療エラーの詳細分類タスク\n\n"
            "以下の臨床症例テキストに含まれる医療エラーを、より詳細なカテゴリに分類してください。\n\n"
            "### 臨床症例テキスト\n"
            "{sentences}\n\n"
            "### エラー情報\n"
            "エラーフラグ: {error_flag}\n"
            "{error_info}\n\n"
            "### エラータイプの分類基準\n\n"
            "以下の8つのカテゴリから**1つだけ**を選択してください：\n\n"
            "1. **病歴聴取エラー** (history_taking): 患者の主訴、現病歴、既往歴、家族歴、社会歴の聴取や記録の誤り\n"
            "2. **身体所見エラー** (physical_findings): 身体診察所見の記録や解釈の誤り\n"
            "3. **検査解釈エラー** (test_interpretation): 血液検査、画像検査、生理機能検査などの結果解釈の誤り\n"
            "4. **診断エラー** (diagnosis): 病名の診断や鑑別診断の誤り\n"
            "5. **薬剤選択エラー** (medication_selection): 薬剤の選択や適応の誤り\n"
            "6. **薬剤用法・用量エラー** (medication_dosage): 薬剤の用法・用量・投与方法の誤り\n"
            "7. **手技・介入エラー** (procedure_intervention): 医療手技、手術、処置の選択や実施の誤り\n"
            "8. **経過観察・管理エラー** (monitoring_management): 治療方針、経過観察、フォローアップの誤り\n\n"
            "### 分類の指針\n\n"
            "- 既存のエラータイプ「{original_error_type}」を参考にしつつ、より詳細な分類を行ってください\n"
            "- エラー文：「{error_sentence}」に注目して分類してください\n"
            "- 複数のカテゴリに該当する場合は、最も主要な問題を選択してください\n"
            "- CORRECTデータ（エラーなし）の場合は「none」を返してください\n\n"
            "### 出力形式\n\n"
            "以下のJSON形式で回答してください：\n\n"
            "{{\n"
            '    "error_type": "[上記8カテゴリのいずれか、またはnone]",\n'
            '    "confidence": "[high/medium/low]",\n'
            '    "explanation": "分類理由の簡潔な説明"\n'
            "}}"
        ),
        "error_type_classification_en": (
            "## Medical Error Detailed Classification Task\n\n"
            "Please classify the medical error contained in the following clinical case text into a more detailed category.\n\n"
            "### Clinical Case Text\n"
            "{sentences}\n\n"
            "### Error Information\n"
            "Error Flag: {error_flag}\n"
            "{error_info}\n\n"
            "### Error Type Classification Criteria\n\n"
            "Select **exactly one** from the following 8 categories:\n\n"
            "1. **History Taking Error** (history_taking): Errors in taking or recording patient's chief complaint, history of present illness, past medical history, family history, or social history\n"
            "2. **Physical Findings Error** (physical_findings): Errors in physical examination findings or their interpretation\n"
            "3. **Test Interpretation Error** (test_interpretation): Errors in interpreting blood tests, imaging studies, physiological function tests, etc.\n"
            "4. **Diagnosis Error** (diagnosis): Errors in disease diagnosis or differential diagnosis\n"
            "5. **Medication Selection Error** (medication_selection): Errors in drug selection or indication\n"
            "6. **Medication Dosage Error** (medication_dosage): Errors in drug dosage, administration route, or frequency\n"
            "7. **Procedure/Intervention Error** (procedure_intervention): Errors in selecting or performing medical procedures, surgeries, or treatments\n"
            "8. **Monitoring/Management Error** (monitoring_management): Errors in treatment planning, follow-up care, or patient monitoring\n\n"
            "### Classification Guidelines\n\n"
            '- Consider the original error type "{original_error_type}" as reference while providing more detailed classification\n'
            '- Focus on the error sentence: "{error_sentence}" for classification\n'
            "- If multiple categories apply, select the most primary issue\n"
            '- For CORRECT data (no errors), return "none"\n\n'
            "### Output Format\n\n"
            "Respond in the following JSON format:\n\n"
            "{{\n"
            '    "error_type": "[one of the 8 categories above, or none]",\n'
            '    "confidence": "[high/medium/low]",\n'
            '    "explanation": "Brief explanation of the classification rationale"\n'
            "}}"
        ),
    }

    def generate_messages(self, sample: MEDECSample) -> List[Dict[str, str]]:
        """Generate conversation messages for MEDEC error type classification."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")

        template_string = self.get_template_string()

        # Determine language based on template name
        is_japanese = self.template_name.endswith("_ja")

        # Format error information
        if sample.error_flag == 1:
            if is_japanese:
                error_info = (
                    f"エラー文番号: {sample.error_sentence_id}\n"
                    f"エラー文: {sample.error_sentence}\n"
                    f"修正文: {sample.corrected_sentence}"
                )
                error_flag_display = "あり"
            else:
                error_info = (
                    f"Error Sentence ID: {sample.error_sentence_id}\n"
                    f"Error Sentence: {sample.error_sentence}\n"
                    f"Corrected Sentence: {sample.corrected_sentence}"
                )
                error_flag_display = "Yes"
        else:
            error_info = "エラーなし" if is_japanese else "No errors"
            error_flag_display = "なし" if is_japanese else "No"

        # Handle original error type (may be None)
        original_error_type = (
            sample.error_type
            if sample.error_type
            else ("不明" if is_japanese else "unknown")
        )
        error_sentence = (
            sample.error_sentence
            if sample.error_sentence
            else ("なし" if is_japanese else "none")
        )

        # Format template
        user_content = template_string.format(
            sentences=sample.sentences,
            error_flag=error_flag_display,
            error_info=error_info,
            original_error_type=original_error_type,
            error_sentence=error_sentence,
        )

        # System prompt
        if is_japanese:
            system_content = (
                "あなたは医療エラー分析の専門家です。"
                "臨床症例テキストに含まれる医療エラーを詳細なカテゴリに分類してください。"
                "医学的知識と臨床経験に基づいて、正確で一貫性のある分類を行ってください。"
            )
        else:
            system_content = (
                "You are a medical error analysis expert. "
                "Please classify medical errors in clinical case texts into detailed categories. "
                "Base your classification on medical knowledge and clinical experience to ensure "
                "accurate and consistent categorization."
            )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
