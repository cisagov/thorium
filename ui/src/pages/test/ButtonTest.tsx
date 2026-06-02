import { useState } from 'react';
import styled from 'styled-components';
import { FaPlus, FaTrash, FaPen, FaCheck, FaDownload, FaMagnifyingGlass, FaCopy, FaArrowRight } from 'react-icons/fa6';

// project imports
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import Subtitle from '@components/shared/titles/Subtitle';
import { Button, IconButton, ButtonVariant, ButtonSize } from '@components/shared/buttons';

const Section = styled.div`
  margin-bottom: 2.5rem;
`;

const SectionLabel = styled.h5`
  color: var(--thorium-secondary-text);
  margin-bottom: 0.75rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const ButtonRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 1rem;
`;

const DarkBg = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--thorium-body-bg);
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid var(--thorium-panel-border);
`;

const PanelBg = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--thorium-panel-bg);
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid var(--thorium-panel-border);
`;

const ClickCount = styled.span`
  font-size: 0.85rem;
  color: var(--thorium-secondary-text);
  min-width: 30px;
`;

const FocusIndicator = styled.div`
  font-size: 0.85rem;
  color: var(--thorium-secondary-text);
  margin-top: 0.5rem;
`;

const ALL_VARIANTS = Object.values(ButtonVariant);
const FILLED_VARIANTS = ALL_VARIANTS.filter((v) => v !== ButtonVariant.Ghost && v !== ButtonVariant.Icon);
const ALL_SIZES = Object.values(ButtonSize);

const variantLabel = (v: ButtonVariant): string => v.charAt(0).toUpperCase() + v.slice(1);
const sizeLabel = (s: ButtonSize): string => {
  const map: Record<ButtonSize, string> = {
    [ButtonSize.XSmall]: 'XSmall',
    [ButtonSize.Small]: 'Small',
    [ButtonSize.Medium]: 'Medium',
    [ButtonSize.Large]: 'Large',
  };
  return map[s];
};

const ButtonTest = () => {
  const [clickCounts, setClickCounts] = useState<Record<string, number>>({});
  const [lastFocused, setLastFocused] = useState('');
  const [disabledClickAttempted, setDisabledClickAttempted] = useState(false);

  const trackClick = (id: string) => {
    setClickCounts((prev) => ({ ...prev, [id]: (prev[id] ?? 0) + 1 }));
  };

  return (
    <Page title="Button Test">
      <Title>Button Component Test</Title>

      <Section data-testid="section-variants">
        <Subtitle>Filled Variants</Subtitle>
        <ButtonRow>
          {FILLED_VARIANTS.map((v) => (
            <Button key={v} variant={v} data-testid={`btn-${v}`}>
              {variantLabel(v)}
            </Button>
          ))}
        </ButtonRow>
      </Section>

      <Section data-testid="section-ghost-icon">
        <Subtitle>Ghost &amp; Icon Variants</Subtitle>
        <SectionLabel>Ghost (transparent background, text-colored)</SectionLabel>
        <ButtonRow>
          <DarkBg>
            <Button variant={ButtonVariant.Ghost} data-testid="btn-ghost">
              Ghost
            </Button>
            <Button variant={ButtonVariant.Ghost} size={ButtonSize.Small} data-testid="btn-ghost-small">
              Small Ghost
            </Button>
          </DarkBg>
          <PanelBg>
            <Button variant={ButtonVariant.Ghost} data-testid="btn-ghost-panel">
              On Panel
            </Button>
          </PanelBg>
        </ButtonRow>

        <SectionLabel>Icon (transparent, color change on hover)</SectionLabel>
        <ButtonRow>
          <DarkBg>
            <IconButton data-testid="iconbtn-pen">
              <FaPen />
            </IconButton>
            <IconButton data-testid="iconbtn-trash">
              <FaTrash />
            </IconButton>
            <IconButton data-testid="iconbtn-search">
              <FaMagnifyingGlass />
            </IconButton>
          </DarkBg>
          <PanelBg>
            <IconButton data-testid="iconbtn-download">
              <FaDownload />
            </IconButton>
            <IconButton $round data-testid="iconbtn-plus-round">
              <FaPlus />
            </IconButton>
          </PanelBg>
        </ButtonRow>
      </Section>

      <Section data-testid="section-sizes">
        <Subtitle>Sizes</Subtitle>
        {ALL_SIZES.map((s) => (
          <div key={s} data-testid={`size-group-${s}`}>
            <SectionLabel>{sizeLabel(s)}</SectionLabel>
            <ButtonRow>
              <Button variant={ButtonVariant.Primary} size={s} data-testid={`btn-primary-${s}`}>
                Primary
              </Button>
              <Button variant={ButtonVariant.Ok} size={s} data-testid={`btn-ok-${s}`}>
                Ok
              </Button>
              <Button variant={ButtonVariant.Danger} size={s} data-testid={`btn-danger-${s}`}>
                Danger
              </Button>
              <Button variant={ButtonVariant.Ghost} size={s} data-testid={`btn-ghost-${s}`}>
                Ghost
              </Button>
              <IconButton size={s} data-testid={`iconbtn-${s}`}>
                <FaPen />
              </IconButton>
            </ButtonRow>
          </div>
        ))}
      </Section>

      <Section data-testid="section-icons">
        <Subtitle>With Icons</Subtitle>
        <SectionLabel>Icon + Text</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Ok} data-testid="btn-icon-create">
            <FaPlus /> Create
          </Button>
          <Button variant={ButtonVariant.Danger} data-testid="btn-icon-delete">
            <FaTrash /> Delete
          </Button>
          <Button variant={ButtonVariant.Primary} data-testid="btn-icon-download">
            <FaDownload /> Download
          </Button>
          <Button variant={ButtonVariant.Warning} data-testid="btn-icon-copy">
            <FaCopy /> Copy
          </Button>
          <Button variant={ButtonVariant.Secondary} data-testid="btn-icon-search">
            <FaMagnifyingGlass /> Search
          </Button>
          <Button variant={ButtonVariant.Info} data-testid="btn-icon-apply">
            <FaCheck /> Apply
          </Button>
        </ButtonRow>

        <SectionLabel>Text + Trailing Icon</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Ok} data-testid="btn-continue">
            Continue <FaArrowRight />
          </Button>
          <Button variant={ButtonVariant.Primary} data-testid="btn-next">
            Next <FaArrowRight />
          </Button>
        </ButtonRow>
      </Section>

      <Section data-testid="section-iconbutton-variants">
        <Subtitle>IconButton Variants</Subtitle>
        <SectionLabel>Default (Icon variant)</SectionLabel>
        <ButtonRow>
          <IconButton data-testid="iconbtn-default-pen">
            <FaPen />
          </IconButton>
          <IconButton data-testid="iconbtn-default-trash">
            <FaTrash />
          </IconButton>
          <IconButton data-testid="iconbtn-default-check">
            <FaCheck />
          </IconButton>
          <IconButton data-testid="iconbtn-default-copy">
            <FaCopy />
          </IconButton>
        </ButtonRow>

        <SectionLabel>Filled Variants</SectionLabel>
        <ButtonRow>
          <IconButton variant={ButtonVariant.Primary} data-testid="iconbtn-filled-primary">
            <FaPen />
          </IconButton>
          <IconButton variant={ButtonVariant.Ok} data-testid="iconbtn-filled-ok">
            <FaCheck />
          </IconButton>
          <IconButton variant={ButtonVariant.Danger} data-testid="iconbtn-filled-danger">
            <FaTrash />
          </IconButton>
          <IconButton variant={ButtonVariant.Warning} data-testid="iconbtn-filled-warning">
            <FaCopy />
          </IconButton>
        </ButtonRow>

        <SectionLabel>Round</SectionLabel>
        <ButtonRow>
          <IconButton $round data-testid="iconbtn-round-default">
            <FaPlus />
          </IconButton>
          <IconButton $round variant={ButtonVariant.Ok} data-testid="iconbtn-round-ok">
            <FaCheck />
          </IconButton>
          <IconButton $round variant={ButtonVariant.Danger} data-testid="iconbtn-round-danger">
            <FaTrash />
          </IconButton>
        </ButtonRow>
      </Section>

      <Section data-testid="section-disabled">
        <Subtitle>Disabled States</Subtitle>
        <SectionLabel>Filled</SectionLabel>
        <ButtonRow>
          {FILLED_VARIANTS.map((v) => (
            <Button key={v} variant={v} disabled data-testid={`btn-disabled-${v}`}>
              {variantLabel(v)}
            </Button>
          ))}
        </ButtonRow>

        <SectionLabel>Ghost &amp; Icon</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Ghost} disabled data-testid="btn-disabled-ghost">
            Ghost
          </Button>
          <IconButton disabled data-testid="iconbtn-disabled-pen">
            <FaPen />
          </IconButton>
          <IconButton disabled data-testid="iconbtn-disabled-trash">
            <FaTrash />
          </IconButton>
        </ButtonRow>

        <SectionLabel>With Icons</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Ok} disabled data-testid="btn-disabled-ok-icon">
            <FaPlus /> Create
          </Button>
          <Button variant={ButtonVariant.Danger} disabled data-testid="btn-disabled-danger-icon">
            <FaTrash /> Delete
          </Button>
        </ButtonRow>
      </Section>

      <Section data-testid="section-clicks">
        <Subtitle>Click Tracking</Subtitle>
        <SectionLabel>Click these buttons and verify the counter increments</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Ok} data-testid="btn-click-ok" onClick={() => trackClick('ok')}>
            Click Me
          </Button>
          <ClickCount data-testid="click-count-ok">{clickCounts['ok'] ?? 0}</ClickCount>

          <Button variant={ButtonVariant.Primary} data-testid="btn-click-primary" onClick={() => trackClick('primary')}>
            Click Me
          </Button>
          <ClickCount data-testid="click-count-primary">{clickCounts['primary'] ?? 0}</ClickCount>

          <IconButton data-testid="btn-click-icon" onClick={() => trackClick('icon')}>
            <FaPen />
          </IconButton>
          <ClickCount data-testid="click-count-icon">{clickCounts['icon'] ?? 0}</ClickCount>
        </ButtonRow>

        <SectionLabel>Disabled buttons should not fire clicks</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Danger} disabled data-testid="btn-click-disabled" onClick={() => setDisabledClickAttempted(true)}>
            Disabled
          </Button>
          <ClickCount data-testid="disabled-click-result">{disabledClickAttempted ? 'FIRED' : 'blocked'}</ClickCount>
        </ButtonRow>
      </Section>

      <Section data-testid="section-tab-nav">
        <Subtitle>Tab Navigation</Subtitle>
        <SectionLabel>Tab through these buttons — focus ring should appear on each</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Primary} data-testid="tab-btn-1" onFocus={() => setLastFocused('tab-btn-1')}>
            First
          </Button>
          <Button variant={ButtonVariant.Ok} data-testid="tab-btn-2" onFocus={() => setLastFocused('tab-btn-2')}>
            Second
          </Button>
          <Button variant={ButtonVariant.Warning} data-testid="tab-btn-3" onFocus={() => setLastFocused('tab-btn-3')}>
            Third
          </Button>
          <Button variant={ButtonVariant.Danger} disabled data-testid="tab-btn-disabled" onFocus={() => setLastFocused('tab-btn-disabled')}>
            Disabled (skip)
          </Button>
          <Button variant={ButtonVariant.Secondary} data-testid="tab-btn-4" onFocus={() => setLastFocused('tab-btn-4')}>
            Fourth
          </Button>
          <IconButton data-testid="tab-btn-5" onFocus={() => setLastFocused('tab-btn-5')}>
            <FaPen />
          </IconButton>
        </ButtonRow>
        <FocusIndicator data-testid="last-focused">Last focused: {lastFocused || 'none'}</FocusIndicator>
      </Section>

      <Section data-testid="section-patterns">
        <Subtitle>Common Patterns</Subtitle>
        <SectionLabel>Modal Footer (Confirm / Cancel)</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Danger} data-testid="pattern-delete">
            Delete
          </Button>
          <Button variant={ButtonVariant.Primary} data-testid="pattern-cancel">
            Cancel
          </Button>
        </ButtonRow>

        <SectionLabel>Form Actions (Save / Discard / Cancel)</SectionLabel>
        <ButtonRow>
          <Button variant={ButtonVariant.Ok} data-testid="pattern-save">
            Save
          </Button>
          <Button variant={ButtonVariant.Warning} data-testid="pattern-discard">
            Discard
          </Button>
          <Button variant={ButtonVariant.Secondary} data-testid="pattern-form-cancel">
            Cancel
          </Button>
        </ButtonRow>

        <SectionLabel>Toolbar (Icon Buttons)</SectionLabel>
        <ButtonRow>
          <IconButton data-testid="pattern-toolbar-pen">
            <FaPen />
          </IconButton>
          <IconButton data-testid="pattern-toolbar-copy">
            <FaCopy />
          </IconButton>
          <IconButton data-testid="pattern-toolbar-trash">
            <FaTrash />
          </IconButton>
          <IconButton data-testid="pattern-toolbar-download">
            <FaDownload />
          </IconButton>
        </ButtonRow>
      </Section>
    </Page>
  );
};

export default ButtonTest;
