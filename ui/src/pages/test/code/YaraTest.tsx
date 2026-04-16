import React from 'react';
import RuleEditorTestPage from './RuleEditorTestPage';
import { YaraRuleChecker } from '@utilities/rules/yara';
import { FormatType } from '@utilities/rules/types';

const SAMPLE_RULE = `import "pe"

rule DetectUPX : packer
{
    meta:
        description = "Detects UPX packed executables"
        author = "Test Author"
        date = "2024-01-15"

    strings:
        $upx0 = "UPX0" ascii
        $upx1 = "UPX1" ascii
        $hex = { 60 E8 00 00 00 00 58 }

    condition:
        uint16(0) == 0x5A4D and ($upx0 or $upx1) and $hex
}`;

const checker = new YaraRuleChecker();

const YaraTest: React.FC = () => (
  <RuleEditorTestPage
    title="YARA Rule Editor Test"
    sampleRule={SAMPLE_RULE}
    checker={checker}
    format={FormatType.YARA}
    helpersKey="__yaraTestHelpers"
  />
);

export default YaraTest;
