"""MEDEC task templates for unified message generation."""

from typing import List, Dict, Any
from .base import BaseTemplate
from ..data.medec.samples import MEDECSample


class MEDECTemplate(BaseTemplate):
    """Template for MEDEC error detection and correction tasks."""

    def __init__(self, template_name: str):
        """Initialize template with name and extract base template information."""
        super().__init__(template_name)

        # Extract base template name and check if it's a lark variant
        if template_name.endswith("_lark"):
            self.base_template_name = template_name[:-5]  # Remove '_lark' suffix
            self.is_lark = True
        else:
            self.base_template_name = template_name
            self.is_lark = False

        # Extract error probability from template name if it's a prior template
        self.error_probability = self._extract_error_probability(template_name)

    # Constants
    LARK_SUFFIX = "_lark"
    PRIOR_PATTERN = r".*_prior_(\d+)_(?:en|ja)$"
    PRIOR_REPLACEMENT = r"_prior_\d+_"

    # Basic shot templates
    BASIC_TEMPLATES = {
        "0_shot_en": (
            "You are a medical expert reviewing clinical text for accuracy. The text contains either no errors or exactly one medical error.\n\n"
            "Identify and correct any medical error related to treatment, diagnosis, management, or causation.\n\n"
            "Output Format:\n"
            "- If no error: `CORRECT`\n"
            "- If error found: `sentence_number: corrected_sentence`\n\n"
            "CRITICAL: Output ONLY the result. Do NOT include explanations, analysis, or additional text.\n\n"
            "{sentences}"
        ),
        "2_shot_en": (
            "# Instructions\n\n"
            "You are a medical expert reviewing clinical text for accuracy. The text contains either no errors or exactly one medical error.\n\n"
            "Identify and correct any medical error related to treatment, diagnosis, management, or causation.\n\n"
            "# Examples\n\n"
            "Example 1 (Correct case):\n"
            "1. A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.\n"
            "2. He works as a commercial fisherman on Lake Superior.\n"
            "3. Current medications include metoprolol and warfarin.\n"
            "4. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg.\n"
            "5. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.\n"
            "6. An x-ray of the chest shows consolidation of the right upper lobe.\n"
            "7. The causal pathogen is Streptococcus pneumoniae.\n"
            "Output: `CORRECT`\n\n"
            "Example 2 (Error case):\n"
            "1. A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.\n"
            "2. He works as a commercial fisherman on Lake Superior.\n"
            "3. Current medications include metoprolol and warfarin.\n"
            "4. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg.\n"
            "5. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.\n"
            "6. After reviewing imaging, the causal pathogen was determined to be Haemophilus influenzae.\n"
            "7. An x-ray of the chest showed consolidation of the right upper lobe.\n"
            "Output: `6: After reviewing imaging, the causal pathogen was determined to be Streptococcus pneumoniae.`\n\n"
            "# Output Format\n\n"
            "Output ONLY `CORRECT` or `number: corrected_sentence`. No explanations.\n\n"
            "{sentences}"
        ),
        "5_shot_en": (
            "# Instructions\n\n"
            "You are a medical expert reviewing clinical text for accuracy. The text contains either no errors or exactly one medical error.\n\n"
            "Identify and correct any medical error related to treatment, diagnosis, management, or causation.\n\n"
            "# Examples\n\n"
            "Example 1 (Correct case):\n"
            "1. A 24-year-old woman comes to the physician because of a 3-day history of fever, headache, myalgia, and nonproductive cough.\n"
            "2. Two weeks ago, she received a parrot as a gift from her boyfriend.\n"
            "3. Her temperature is 38.5 C (101.3 F), pulse is 112/min, respirations are 22/min, and blood pressure is 100/70 mm Hg.\n"
            "4. Pulmonary examination shows fine crackles in the lower lung fields bilaterally.\n"
            "5. Laboratory studies show a leukocyte count of 6,800/mm3.\n"
            "6. An x-ray of the chest shows bilateral lower lobe infiltrates.\n"
            "7. Chlamydophila psittaci infection is diagnosed.\n"
            "Output: `CORRECT`\n\n"
            "Example 2 (Pathogen error):\n"
            "1. A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.\n"
            "2. He works as a commercial fisherman on Lake Superior.\n"
            "3. Current medications include metoprolol and warfarin.\n"
            "4. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg.\n"
            "5. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.\n"
            "6. After reviewing imaging, the causal pathogen was determined to be Haemophilus influenzae.\n"
            "7. An x-ray of the chest showed consolidation of the right upper lobe.\n"
            "Output: `6: After reviewing imaging, the causal pathogen was determined to be Streptococcus pneumoniae.`\n\n"
            "Example 3 (Management error):\n"
            "1. A 1-year-old boy is brought to the physician because of fever for 3 days.\n"
            "2. He was treated with intravenous ceftriaxone and became afebrile within 24 hours.\n"
            "3. Urine culture shows Escherichia coli that is sensitive to ceftriaxone.\n"
            "4. This is his first episode of urinary tract infection and there is no family history of urological abnormalities.\n"
            "5. Cystourethrogram is voided.\n"
            "Output: `5: Renal bladder ultrasound is recommended.`\n\n"
            "Example 4 (Correct case):\n"
            "1. A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum.\n"
            "2. He works as a commercial fisherman on Lake Superior.\n"
            "3. Current medications include metoprolol and warfarin.\n"
            "4. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg.\n"
            "5. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field.\n"
            "6. An x-ray of the chest shows consolidation of the right upper lobe.\n"
            "7. The causal pathogen is Streptococcus pneumoniae.\n"
            "Output: `CORRECT`\n\n"
            "Example 5 (Diagnosis error):\n"
            "1. A 42-year-old man with ulcerative colitis comes to the emergency department because of bloody diarrhea and abdominal pain for 2 days.\n"
            "2. Current medications include 5-aminosalicylic acid.\n"
            "3. He appears ill and has a distended, tense abdomen.\n"
            "4. Abdominal examination is limited because of severe pain.\n"
            "5. His temperature is 38.9 C (102 F), pulse is 120/min, and blood pressure is 90/60 mm Hg.\n"
            "6. Patient is diagnosed with perforated diverticulitis.\n"
            "Output: `6: Patient is diagnosed with toxic megacolon.`\n\n"
            "# Output Format\n\n"
            "Output ONLY `CORRECT` or `number: corrected_sentence`. No explanations.\n\n"
            "{sentences}"
        ),
        "0_shot_ja": (
            "あなたは臨床テキストの正確性をレビューする医学専門家です。テキストにはエラーがないか、または正確に1つの医学的エラーが含まれています。\n\n"
            "治療、診断、管理、または因果関係に関連する医学的エラーを特定し、修正してください。\n\n"
            "出力形式：\n"
            "- エラーなしの場合：`CORRECT`\n"
            "- エラー発見の場合：`文番号: 修正された文`\n\n"
            "重要：結果のみを出力してください。説明、分析、追加のテキストは含めないでください。\n\n"
            "{sentences}"
        ),
        "2_shot_ja": (
            "# 指示\n\n"
            "あなたは臨床テキストの正確性をレビューする医学専門家です。テキストにはエラーがないか、または正確に1つの医学的エラーが含まれています。\n\n"
            "治療、診断、管理、または因果関係に関連する医学的エラーを特定し、修正してください。\n\n"
            "# 具体例\n\n"
            "例1（正しいケース）：\n"
            # Sample ID: 119A16_d_deepseek-r1-0528
            "1. 10歳の女子。感冒時に実施された血液検査で肝障害を指摘され紹介受診した。自覚症状はない。\n"
            "2. 身長137 cm、体重36 kg。\n"
            "3. 体温36.8 °C。脈拍76/分、整。血圧104/70 mmHg。\n"
            "4. 眼疑結膜と眼球結膜とに異常を認めない。\n"
            "5. 細隙灯顕微鏡検査にてKayser-Fleischer輪を確認した。\n"
            "6. 頸部リンパ節を触知しない。\n"
            "7. 腹部は平坦で軟、右肋骨弓下に肝を1 cm触知する。脾は触知しない。\n"
            "8. 尿所見: 蛋白(-)、糖(-)、潜血(-)。尿中銅排泄量200 μg/日（基準80未満）。\n"
            "9. 血液所見: 赤血球409万、Hb 12.1 g/dL、白血球8,100、血小板33万。\n"
            "10. 血液生化学所見: AST 156 U/L、ALT 245 U/L、LD 308 U/L（基準145~270）、銅25 μg/dL（基準68~128）、セルロプラスミン12 mg/dL（基準21~37）。\n"
            "11. 免疫血清学所見: CRP 0.1 mg/dL、HBs抗原陰性、HCV抗体陰性、抗EBV VCA IgM抗体陰性、抗EBV VCA IgG抗体陰性。\n"
            "出力：`CORRECT`\n\n"
            "例2（エラーケース）：\n"
            # Sample ID: 119A16_a_deepseek-r1-0528
            "1. 10歳の女子。感冒時に実施された血液検査で肝障害を指摘され紹介受診した。自覚症状はない。\n"
            "2. 身長137 cm、体重36 kg。\n"
            "3. 体温36.8 °C。脈拍76/分、整。血圧104/70 mmHg。\n"
            "4. 眼瞼結膜と眼球結膜とに異常を認めない。\n"
            "5. 角膜形状検査で両側性の円錐角膜を認めた。\n"
            "6. 頸部リンパ節を触知しない。\n"
            "7. 腹部は平坦で軟、右肋骨弓下に肝を1 cm触知する。脾は触知しない。\n"
            "8. 尿所見: 蛋白(-)、糖(-)、潜血(-)。尿中銅排泄量200 μg/日（基準80未満）。\n"
            "9. 血液所見: 赤血球409万、Hb 12.1 g/dL、白血球8,100、血小板33万。\n"
            "10. 血液生化学所見: AST 156 U/L、ALT 245 U/L、LD 308 U/L（基準145~270）、銅25 μg/dL（基準68~128）、セルロプラスミン12 mg/dL（基準21~37）。\n"
            "11. 免疫血清学所見: CRP 0.1 mg/dL、HBs抗原陰性、HCV抗体陰性、抗EBV VCA IgM抗体陰性、抗EBV VCA IgG抗体陰性。\n"
            "出力：`5: 細隙灯顕微鏡検査にてKayser-Fleischer輪を確認した。`\n\n"
            "# 出力形式\n\n"
            "`CORRECT`または`番号: 修正された文`のどちらかの形式で出力してください。追加の説明は不要です。\n\n"
            "{sentences}"
        ),
        "5_shot_ja": (
            "# 指示\n\n"
            "あなたは臨床テキストの正確性をレビューする医学専門家です。テキストにはエラーがないか、または正確に1つの医学的エラーが含まれています。\n\n"
            "治療、診断、管理、または因果関係に関連する医学的エラーを特定し、修正してください。\n\n"
            "# 具体例\n\n"
            "例1（正しいケース）：\n"
            # Sample ID: 119A25_e_deepseek-r1-0528
            "1. 日齢20の男児が哺乳量低下と発熱を主訴に母親に連れられ来院した。\n"
            "2. 在胎39週3日、体重3,120gで出生した。\n"
            "3. 昨日から哺乳量減少が持続し、本日38.6°Cの発熱が認められた。\n"
            "4. 身体所見：顔色不良、大泉門膨隆、易刺激性を認めた。\n"
            "5. 血液検査：赤血球412万、Hb 12.1g/dL、Ht 36％、白血球25,000（桿状核15％/分葉核65％/単球10％/リンパ球10％）、血小板15万。\n"
            "6. 生化学：血糖98mg/dL、Na 136mEq/L、K 4.5mEq/L、Cl 100mEq/L、CRP 13.8mg/dL。\n"
            "7. 脳脊髄液：細胞数4,200/mm³（単核球22％/多形核球78％）、蛋白80mg/dL、糖5mg/dL。\n"
            "8. 検査所見からGBS（Streptococcus agalactiae）起因の髄膜炎が強く示唆された。\n"
            "出力：`CORRECT`\n\n"
            "例2（エラーケース）：\n"
            # Sample ID: 119A30_e_deepseek-r1-0528
            "1. 64歳女性、空腹時の動悸と発汗を主訴に来院。\n"
            "2. 1か月前から朝食後外出時、昼食前に空腹感と共に動悸・発汗・手指振戦出現。\n"
            "3. 症状は甘い物摂取で改善。\n"
            "4. 慢性甲状腺炎に伴う甲状腺ホルモン不安定性が代謝異常を引き起こし症状発現と診断。\n"
            "5. 既往歴：脂質異常症、耐糖能異常、慢性甲状腺炎、胆石症。\n"
            "6. 現内服薬：スタチン（脂質異常症治療）。\n"
            "7. 身体所見：身長156cm、体重62kg、体温36.2°C、脈拍72/分整、血圧142/88mmHg。\n"
            "8. 眼瞼結膜・眼球結膜異常なし、甲状腺非触知、心音・呼吸音正常、腹部平坦軟、肝脾触知せず。\n"
            "9. 検査所見：AST 28U/L、ALT 32U/L、γ-GT 72U/L、血糖110mg/dL、HbA1c 6.1%、総コレステロール182mg/dL、TG 180mg/dL、HDL 38mg/dL、TSH 1.2μU/mL、FT4 1.4ng/dL。\n"
            "出力：`4: 慢性甲状腺炎の既往があるが、TSH/FT4正常値から現時点での甲状腺機能異常は認められないと判断。`\n\n"
            "例3（エラーケース）：\n"
            # Sample ID: 119A16_a_deepseek-r1-0528
            "1. 10歳の女子。感冒時に実施された血液検査で肝障害を指摘され紹介受診した。自覚症状はない。\n"
            "2. 身長137 cm、体重36 kg。\n"
            "3. 体温36.8 °C。脈拍76/分、整。血圧104/70 mmHg。\n"
            "4. 眼瞼結膜と眼球結膜とに異常を認めない。\n"
            "5. 角膜形状検査で両側性の円錐角膜を認めた。\n"
            "6. 頸部リンパ節を触知しない。\n"
            "7. 腹部は平坦で軟、右肋骨弓下に肝を1 cm触知する。脾は触知しない。\n"
            "8. 尿所見: 蛋白(-)、糖(-)、潜血(-)。尿中銅排泄量200 μg/日（基準80未満）。\n"
            "9. 血液所見: 赤血球409万、Hb 12.1 g/dL、白血球8,100、血小板33万。\n"
            "10. 血液生化学所見: AST 156 U/L、ALT 245 U/L、LD 308 U/L（基準145~270）、銅25 μg/dL（基準68~128）、セルロプラスミン12 mg/dL（基準21~37）。\n"
            "11. 免疫血清学所見: CRP 0.1 mg/dL、HBs抗原陰性、HCV抗体陰性、抗EBV VCA IgM抗体陰性、抗EBV VCA IgG抗体陰性。\n"
            "出力：`5: 細隙灯顕微鏡検査にてKayser-Fleischer輪を確認した。`\n\n"
            "例4（エラーケース）：\n"
            # Sample ID: 119A19_a_deepseek-r1-0528
            "1. 52歳の女性。健康診断の胸部X線写真で異常を指摘され来院した。\n"
            "2. 咳嗽が3か月前から出現していたが医療機関を受診していなかった。\n"
            "3. 既往歴に特記すべきことはない。\n"
            "4. 職業は小学校教員である。\n"
            "5. 胸部単純CTで右肺上葉に気管支拡張病変と空洞を認めた。\n"
            "6. 患者は喀痰検体を提出し帰宅した。\n"
            "7. 同日の夕方、細菌検査室から喀痰抗酸菌染色が陽性であると医師に報告があった。\n"
            "8. 医師は勤務先の小学校に直接連絡し同僚への感染リスクを伝達した。\n"
            "9. 患者には自宅で健康観察を継続するよう指示した。\n"
            "出力：`8: 医師は患者に電話で自宅待機を指示し、隔離目的で外来再受診を手配した。`\n\n"
            "例5（正しいケース）：\n"
            # Sample ID: 119A16_d_deepseek-r1-0528
            "1. 10歳の女子。感冒時に実施された血液検査で肝障害を指摘され紹介受診した。自覚症状はない。\n"
            "2. 身長137 cm、体重36 kg。\n"
            "3. 体温36.8 °C。脈拍76/分、整。血圧104/70 mmHg。\n"
            "4. 眼瞼結膜と眼球結膜とに異常を認めない。\n"
            "5. 細隙灯顕微鏡検査にてKayser-Fleischer輪を確認した。\n"
            "6. 頸部リンパ節を触知しない。\n"
            "7. 腹部は平坦で軟、右肋骨弓下に肝を1 cm触知する。脾は触知しない。\n"
            "8. 尿所見: 蛋白(-)、糖(-)、潜血(-)。尿中銅排泄量200 μg/日（基準80未満）。\n"
            "9. 血液所見: 赤血球409万、Hb 12.1 g/dL、白血球8,100、血小板33万。\n"
            "10. 血液生化学所見: AST 156 U/L、ALT 245 U/L、LD 308 U/L（基準145~270）、銅25 μg/dL（基準68~128）、セルロプラスミン12 mg/dL（基準21~37）。\n"
            "11. 免疫血清学所見: CRP 0.1 mg/dL、HBs抗原陰性、HCV抗体陰性、抗EBV VCA IgM抗体陰性、抗EBV VCA IgG抗体陰性。\n"
            "出力：`CORRECT`\n\n"
            "# 出力形式\n\n"
            "`CORRECT`または`番号: 修正された文`のどちらかの形式で出力してください。追加の説明は不要です。\n\n"
            "{sentences}"
        ),
    }

    # Prior templates with error probability
    PRIOR_TEMPLATES = {
        "0_shot_prior_en": (
            "You are a medical expert reviewing clinical text for accuracy. Based on the dataset characteristics, approximately {error_probability}% of texts contain exactly one medical error, while {correct_probability}% contain no errors.\n\n"
            "Identify and correct any medical error related to treatment, diagnosis, management, or causation.\n\n"
            "Output Format:\n"
            "- If no error: `CORRECT`\n"
            "- If error found: `sentence_number: corrected_sentence`\n\n"
            "CRITICAL: Output ONLY the result. Do NOT include explanations, analysis, or additional text.\n\n"
            "{sentences}"
        ),
        "0_shot_prior_ja": (
            "あなたは臨床テキストの正確性をレビューする医学専門家です。データセット特性に基づくと、約{error_probability}%のテキストに正確に1つの医学的エラーが含まれ、{correct_probability}%にはエラーがありません。\n\n"
            "治療、診断、管理、または因果関係に関連する医学的エラーを特定し、修正してください。\n\n"
            "出力形式：\n"
            "- エラーなしの場合：`CORRECT`\n"
            "- エラー発見の場合：`文番号: 修正された文`\n\n"
            "重要：結果のみを出力してください。説明、分析、追加のテキストは含めないでください。\n\n"
            "{sentences}"
        ),
    }

    # Reasoning templates with step-by-step analysis
    REASONING_TEMPLATES = {
        "0_shot_reasoning_en": (
            "You are a medical expert reviewing clinical text for accuracy. The text contains either no errors or exactly one medical error.\n\n"
            "Identify and correct any medical error related to treatment, diagnosis, management, or causation.\n\n"
            "First, carefully analyze the text following these steps:\n"
            "1. Verify each sentence based on medical knowledge\n"
            "2. Check consistency between symptoms, test results, and diagnosis\n"
            "3. Evaluate appropriateness of treatment or management\n"
            "4. If an error is found, clearly state the rationale\n\n"
            "After your analysis, output only the final result in the following format:\n"
            "- If no error: `CORRECT`\n"
            "- If error found: `sentence_number: corrected_sentence`\n\n"
            "{sentences}"
        ),
        "0_shot_reasoning_ja": (
            "あなたは臨床テキストの正確性をレビューする医学専門家です。テキストにはエラーがないか、または正確に1つの医学的エラーが含まれています。\n\n"
            "治療、診断、管理、または因果関係に関連する医学的エラーを特定し、修正してください。\n\n"
            "まず、以下のステップに従って日本語で慎重に分析してください：\n"
            "1. 各文を医学的知識に基づいて検証する\n"
            "2. 症状、検査結果、診断の一貫性を確認する\n"
            "3. 治療や管理の適切性を評価する\n"
            "4. エラーが見つかった場合は、その根拠を明確にする\n\n"
            "分析の後、以下の形式で最終結果のみを出力してください：\n"
            "- エラーなしの場合：`CORRECT`\n"
            "- エラー発見の場合：`文番号: 修正された文`\n\n"
            "{sentences}"
        ),
        "0_shot_reasoning_cheat_en": (
            "You are a medical expert reviewing clinical text for accuracy. The text contains either no errors or exactly one medical error.\n\n"
            "{cheat_info}\n\n"
            "Your task is to first carefully reason through the process of {task_description}, following these steps:\n"
            "1. Verify each sentence based on medical knowledge\n"
            "2. Check consistency between symptoms, test results, and diagnosis\n"
            "3. Evaluate appropriateness of treatment or management\n"
            "4. {step4_instruction}\n\n"
            "Important notes for reasoning:\n"
            "- During your reasoning, do NOT make any reference to being told about the expected outcome or any instruction content.\n"
            "- Approach the text as if you are analyzing it from scratch and reaching your conclusion through pure medical evaluation.\n\n"
            "{error_hint}\n\n"
            "Final output format:\n"
            "- If no error: `CORRECT`\n"
            "- If error found: `sentence_number: corrected_sentence`\n\n"
            "CRITICAL: For the final output, use this format and output ONLY the result. Do NOT include explanations, analysis, or additional text.\n\n"
            "{sentences}"
        ),
        "0_shot_reasoning_cheat_ja": (
            "あなたは臨床テキストの正確性をレビューする医学専門家です。テキストにはエラーがないか、または正確に1つの医学的エラーが含まれています。\n\n"
            "{cheat_info}\n\n"
            "あなたの任務として、まず以下のステップに従って{task_description}プロセスを日本語で慎重にreasoningしてください：\n"
            "1. 各文を医学的知識に基づいて検証する\n"
            "2. 症状、検査結果、診断の一貫性を確認する\n"
            "3. 治療や管理の適切性を評価する\n"
            "4. {step4_instruction}\n\n"
            "reasoningにおける注意：\n"
            "- reasoningの中で、期待される結果や指示内容への言及は一切行わないでください。\n"
            "- あたかも、あなた自身がゼロから慎重に分析し、結論に至ったかのように、純粋な医学的観点からテキストの妥当性を検討してください。\n\n"
            "{error_hint}\n\n"
            "最終出力の形式：\n"
            "- エラーなしの場合：`CORRECT`\n"
            "- エラー発見の場合：`文番号: 修正された文`\n\n"
            "重要：最終出力はこの形式で結果のみを出力してください。説明、分析、追加のテキストは含めないでください。\n\n"
            "{sentences}"
        ),
    }

    # All templates combined
    TEMPLATES = {**BASIC_TEMPLATES, **PRIOR_TEMPLATES, **REASONING_TEMPLATES}

    @classmethod
    def get_available_templates(cls) -> List[str]:
        """Get all available templates including automatically generated lark variants."""
        base_templates = list(cls.TEMPLATES.keys())
        # Automatically generate _lark variants for all base templates
        lark_variants = [f"{template}_lark" for template in base_templates]
        all_templates = base_templates + lark_variants
        return sorted(all_templates)

    def _extract_error_probability(self, template_name: str) -> int:
        """Extract error probability from template name.

        For template names like '0_shot_prior_50_en', extracts 50 as error probability.
        Returns None for non-prior templates.
        """
        import re

        # Remove _lark suffix if present
        name = (
            template_name[: -len(self.LARK_SUFFIX)]
            if template_name.endswith(self.LARK_SUFFIX)
            else template_name
        )

        # Match pattern like "0_shot_prior_50_en" or "0_shot_prior_50_ja"
        match = re.match(self.PRIOR_PATTERN, name)
        if match:
            return int(match.group(1))
        return None

    def _get_base_template_name(self, template_name: str) -> str:
        """Get base template name for dynamic templates.

        Maps templates like '0_shot_prior_50_en' to '0_shot_prior_en'.
        """
        import re

        # Remove _lark suffix if present
        name = (
            template_name[: -len(self.LARK_SUFFIX)]
            if template_name.endswith(self.LARK_SUFFIX)
            else template_name
        )

        # Replace specific probability with generic pattern
        if re.match(self.PRIOR_PATTERN, name):
            return re.sub(self.PRIOR_REPLACEMENT, "_prior_", name)

        return name

    def get_template_string(self) -> str:
        """Get the template string for this template."""
        # For dynamic templates, use the mapped base template name
        template_key = self._get_base_template_name(self.base_template_name)

        if template_key not in self.TEMPLATES:
            available = list(self.TEMPLATES.keys())
            raise ValueError(
                f"Template '{template_key}' not found. Available: {available}"
            )

        return self.TEMPLATES[template_key]

    def generate_messages(self, sample: MEDECSample) -> List[Dict[str, str]]:
        """Generate conversation messages for MEDEC sample data."""
        if not isinstance(sample, MEDECSample):
            raise ValueError(f"Expected MEDECSample, got {type(sample)}")

        template_string = self.get_template_string()
        template_params = self._prepare_template_params(sample)
        user_content = self.format_template(template_string, **template_params)

        # Create base messages
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        # Add Lark tools if this is a lark template
        if self.is_lark:
            messages[0].update(self._build_lark_tools())

        return messages

    def _prepare_template_params(self, sample: MEDECSample) -> Dict[str, Any]:
        """Prepare template parameters for formatting."""
        template_params = {
            "sentences": sample.sentences,
            "error_flag": sample.error_flag,
            "error_sentence_id": sample.error_sentence_id,
            "error_sentence": sample.error_sentence,
            "corrected_sentence": sample.corrected_sentence,
        }

        self._add_prior_params(template_params)
        self._add_cheat_params(template_params, sample)

        return template_params

    def _add_prior_params(self, template_params: Dict[str, Any]) -> None:
        """Add prior probability parameters."""
        if self.error_probability is not None:
            template_params.update(
                {
                    "error_probability": self.error_probability,
                    "correct_probability": 100 - self.error_probability,
                }
            )

    def _add_cheat_params(
        self, template_params: Dict[str, Any], sample: MEDECSample
    ) -> None:
        """Add cheat parameters for cheat templates."""
        if "cheat" in self.base_template_name:
            if self.base_template_name.endswith(
                "_ja"
            ) or self.base_template_name.endswith("_cheat_ja"):
                self._add_cheat_params_ja(template_params, sample)
            else:
                self._add_cheat_params_en(template_params, sample)

    def _add_cheat_params_en(
        self, template_params: Dict[str, Any], sample: MEDECSample
    ) -> None:
        """Add cheat parameters for English templates."""
        if sample.error_flag == 1:  # Error example
            template_params.update(
                {
                    "cheat_info": 'This time, you are reviewing an "error example" that has been pre-verified to contain exactly one medical error.',
                    "task_description": "identifying the error",
                    "step4_instruction": "If an error is found, clearly state the rationale and provide correction",
                    "error_hint": f"Reference information (please do not mention in reasoning):\nError sentence: {sample.error_sentence}\nCorrected sentence: {sample.corrected_sentence}",
                }
            )
        else:  # Correct example
            template_params.update(
                {
                    "cheat_info": 'This time, you are reviewing a "no-error example" that has been pre-verified to contain no errors.',
                    "task_description": "confirming that there are no errors",
                    "step4_instruction": "If an error is found, clearly state the rationale",
                    "error_hint": "",
                }
            )

    def _add_cheat_params_ja(
        self, template_params: Dict[str, Any], sample: MEDECSample
    ) -> None:
        """Add cheat parameters for Japanese templates."""
        if sample.error_flag == 1:  # Error example
            template_params.update(
                {
                    "cheat_info": "今回は、正確に1つの医学的エラーが含まれることが事前に確認されている「エラー例」をレビューします。",
                    "task_description": "エラーを特定する",
                    "step4_instruction": "エラーが見つかった場合は、その根拠を明確にし、修正を提供する",
                    "error_hint": f"参考情報（reasoningでは言及しないでください）：\nエラー文：{sample.error_sentence}\n修正文：{sample.corrected_sentence}",
                }
            )
        else:  # Correct example
            template_params.update(
                {
                    "cheat_info": "今回は、エラーがないことが事前に確認されている「エラーなし例」をレビューします。",
                    "task_description": "エラーがないことを確かめる",
                    "step4_instruction": "エラーが見つかった場合は、その根拠を明確にする",
                    "error_hint": "",
                }
            )

    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt based on template type."""
        if self.base_template_name.endswith(
            "_reasoning_ja"
        ) or self.base_template_name.endswith("_reasoning_cheat_ja"):
            return "あなたは臨床テキストの正確性をレビューし、必要に応じて修正を提供する熟練した医師です。推論や思考プロセスはすべて日本語で行ってください。"
        elif self.base_template_name.endswith(
            "_reasoning_en"
        ) or self.base_template_name.endswith("_reasoning_cheat_en"):
            return "You are a skilled medical doctor reviewing clinical text for accuracy and providing corrections when necessary. Please provide your reasoning and thought process."
        elif self.base_template_name.endswith("_ja"):
            return "あなたは臨床テキストの正確性をレビューし、必要に応じて修正を提供する熟練した医師です。"
        else:
            return "You are a skilled medical doctor reviewing clinical text for accuracy and providing corrections when necessary."

    def _build_lark_tools(self) -> Dict[str, Any]:
        """Build Lark tool configuration."""
        return {
            "tools": [
                {
                    "type": "custom",
                    "name": "lark_tool",
                    "custom": {
                        "name": "lark_tool",
                        "format": {
                            "type": "grammar",
                            "grammar": {
                                "syntax": "lark",
                                "definition": 'start: CORRECT | ERROR_RESPONSE\nCORRECT: "CORRECT"\nERROR_RESPONSE: NUMBER ":" SPACE CORRECTION\nNUMBER: /[0-9]+/\nSPACE: " "\nCORRECTION: /[^\\n]+/',
                            },
                        },
                    },
                }
            ],
            "tool_choice": "required",
        }
