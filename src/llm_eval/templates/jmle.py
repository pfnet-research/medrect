"""JMLE data synthesis templates."""

from typing import List, Dict
from .base import BaseTemplate
from ..data.jmle.samples import JMLESample


class JMLETemplate(BaseTemplate):
    """Template for JMLE data synthesis tasks."""

    # Template definitions for clinical narrative generation
    TEMPLATES = {
        "simple_choice_synthesis": (
            "以下の日本の医師国家試験問題を、MEDEC（Medical Error Detection and Correction）形式の臨床症例に変換してください。\n\n"
            "# 指示\n"
            "- 各選択肢について、その選択肢を問題文に組み込んだ臨床記録を1つずつ、全部で5つの記録を合成してください。\n"
            "- その選択肢が正答選択肢の場合は正しい記録、誤答選択肢の場合はエラーを含む記録を合成してほしいです。\n"
            "- 臨床記録は常にMarkdownの番号付きリストの形で書かれ、エラーを含む記録においては1文だけが臨床的なエラーを含むようにします。\n"
            "- 合成した臨床記録は、MEDEC（Medical Error Detection and Correction）形式のベンチマークに使います。エラーを含む記録中にどこがエラーであるかを示さないでください。\n"
            "- 元の問題文に含まれる数値や所見などの情報は、要約したり省略したりせず、可能な限りすべて臨床記録に含めてください。**\n"
            "- 元の問題文や選択肢にない独自の医学的解釈を加えないでください。\n\n"
            "# 元の医師国家試験問題\n"
            "問題文: {question}\n"
            "選択肢:\n"
            "{choices_text}\n"
            "正答選択肢: {correct_choices_list}\n"
            "誤答選択肢: {wrong_choices_list}\n\n"
            "# 合成の形式\n"
            "以下に、正答選択肢から合成した正しい記録と、誤答選択肢から合成したエラー記録の例を示します。\n\n"
            "### 選択肢{correct_choices_list[0]}から合成した正しい記録\n"
            "1. ...\n"
            "2. ...\n"
            "...\n"
            "N. ...\n\n"
            "### 選択肢{wrong_choices_list[0]}から合成したエラー記録\n"
            "1. ...\n"
            "2. ...\n"
            "...\n"
            "N. ...\n"
            "エラータイプ: [病歴聴取エラー, 身体所見エラー, 検査解釈エラー, 診断エラー, 薬剤選択エラー, 薬剤用法・用量エラー, 手技・介入エラー, 経過観察・管理エラー から1つ選択]\n"
            "エラー文番号: [1文だけ、誤った内容を含む文の番号]\n"
            "エラー文: [1文だけ、誤った内容を含む文]\n"
            "修正文: [エラー文を臨床的に正しくした文]\n\n"
            "# 合成例\n\n"
            "### 選択肢dから合成した正しい記録\n"
            "1. 患者は10歳の女子で、感冒時に実施された血液検査で肝障害を指摘され紹介受診した。自覚症状はない。\n"
            "2. 身長137 cm、体重36 kg。\n"
            "3. 体温36.8 °C。脈拍76/分、整。血圧104/70 mmHg。\n"
            "4. 眼瞼結膜と眼球結膜とに異常を認めない。\n"
            "5. 細隙灯顕微鏡検査でKayser-Fleischer輪を認めた。\n"
            "6. 頸部リンパ節を触知しない。\n"
            "7. 腹部は平坦、軟で、右肋骨弓下に肝を1 cm触知する。脾は触知しない。\n"
            "8. 尿所見: 蛋白(-)、糖(-)、潜血(-)。尿中銅排泄量200 μg/日（基準80未満）。\n"
            "9. 血液所見: 赤血球409万、Hb 12.1 g/dL、白血球8,100、血小板33万。\n"
            "10. 血液生化学所見: AST 156 U/L、ALT 245 U/L、LD 308 U/L（基準145~270）、銅25 μg/dL（基準68~128）、セルロプラスミン12 mg/dL（基準21~37）。\n"
            "11. 免疫血清学所見: CRP 0.1 mg/dL、HBs抗原陰性、HCV抗体陰性、抗EBV VCA IgM抗体陰性、抗EBV VCA IgG抗体陰性。\n\n"
            "### 選択肢aから合成したエラー記録\n"
            "1. 患者は10歳の女子で、感冒時に実施された血液検査で肝障害を指摘され紹介受診した。自覚症状はない。\n"
            "2. 身長137 cm、体重36 kg。\n"
            "3. 体温36.8 °C。脈拍76/分、整。血圧104/70 mmHg。\n"
            "4. 眼瞼結膜と眼球結膜とに異常を認めない。\n"
            "5. 眼底検査で典型的な桜実紅斑を認めた。\n"
            "6. 頸部リンパ節を触知しない。\n"
            "7. 腹部は平坦、軟で、右肋骨弓下に肝を1 cm触知する。脾は触知しない。\n"
            "8. 尿所見: 蛋白(-)、糖(-)、潜血(-)。尿中銅排泄量200 μg/日（基準80未満）。\n"
            "9. 血液所見: 赤血球409万、Hb 12.1 g/dL、白血球8,100、血小板33万。\n"
            "10. 血液生化学所見: AST 156 U/L、ALT 245 U/L、LD 308 U/L（基準145~270）、銅25 μg/dL（基準68~128）、セルロプラスミン12 mg/dL（基準21~37）。\n"
            "11. 免疫血清学所見: CRP 0.1 mg/dL、HBs抗原陰性、HCV抗体陰性、抗EBV VCA IgM抗体陰性、抗EBV VCA IgG抗体陰性。\n\n"
            "エラータイプ: 身体所見エラー\n"
            "エラー文番号: 5\n"
            "エラー文: 眼底検査で典型的な桜実紅斑を認めた。\n"
            "修正文: 細隙灯顕微鏡検査でKayser-Fleischer輪を認めた。\n\n"
        ),
    }

    def generate_messages(self, sample: JMLESample) -> List[Dict[str, str]]:
        """Generate conversation messages for JMLE sample data."""
        if not isinstance(sample, JMLESample):
            raise ValueError(f"Expected JMLESample, got {type(sample)}")

        template_string = self.get_template_string()

        # Prepare template variables
        choices_text = "\n".join(
            f"{key}: {value}" for key, value in sample.choices.items()
        )
        correct_choices_list = ", ".join(sample.answer)
        wrong_choices = [
            choice
            for choice in ["a", "b", "c", "d", "e"]
            if choice not in sample.answer
        ]
        wrong_choices_list = ", ".join(wrong_choices)

        user_content = template_string.format(
            question=sample.question,
            choices_text=choices_text,
            correct_choices_list=correct_choices_list,
            wrong_choices_list=wrong_choices_list,
        )

        # Return messages in OpenAI format
        return [
            {
                "role": "system",
                "content": "あなたは医学的知識を持つ専門家です。日本の医師国家試験問題を基に、実際の臨床症例のような自然な医療記録を作成してください。指示に従って、正確に1つのエラーのみを指定された位置に配置してください。",
            },
            {"role": "user", "content": user_content},
        ]
