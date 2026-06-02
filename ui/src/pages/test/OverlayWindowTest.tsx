import { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

// project imports
import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import Subtitle from '@components/shared/titles/Subtitle';
import { OverlayWindow } from '@components/shared/windows/OverlayWindow';
import { Placement, PositionType } from '@components/shared/windows';

const Section = styled.div`
  margin-bottom: 2rem;
`;

const AnchorButton = styled.button`
  padding: 8px 16px;
  border: 1px solid var(--thorium-panel-border);
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  border-radius: 4px;
  cursor: pointer;
  margin-right: 8px;
`;

const WindowContent = styled.div`
  padding: 8px;
  font-size: 14px;
`;

const StatusBar = styled.div`
  margin-top: 1rem;
  padding: 8px;
  background: var(--thorium-secondary-panel-bg);
  border-radius: 4px;
  font-size: 13px;
  font-family: monospace;
`;

const ScrollMarker = styled.div`
  padding: 12px 16px;
  border: 1px dashed var(--thorium-panel-border);
  border-radius: 4px;
  color: var(--thorium-secondary-text);
  font-size: 13px;
  text-align: center;
`;

const Spacer = styled.div<{ $height: number }>`
  height: ${(props) => props.$height}px;
  border-left: 2px dashed var(--thorium-panel-border);
  margin-left: 50%;
  position: relative;

  &::after {
    content: attr(data-label);
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--thorium-secondary-text);
    font-size: 12px;
    white-space: nowrap;
  }
`;

interface WindowVisibility {
  fixedCenter: boolean;
  fixedBottomRight: boolean;
  fixedCustom: boolean;
  absoluteRef: boolean;
  absoluteCustom: boolean;
  absoluteMidpage: boolean;
  fixedNoDrag: boolean;
  headerlessMenu: boolean;
  headerlessNoClose: boolean;
  headerlessOnClose: boolean;
}

const OverlayWindowTest = () => {
  const [visibility, setVisibility] = useState<WindowVisibility>({
    fixedCenter: false,
    fixedBottomRight: false,
    fixedCustom: false,
    absoluteRef: false,
    absoluteCustom: false,
    absoluteMidpage: false,
    fixedNoDrag: false,
    headerlessMenu: false,
    headerlessNoClose: false,
    headerlessOnClose: false,
  });

  const [closedWindows, setClosedWindows] = useState<string[]>([]);

  const anchorRef = useRef<HTMLButtonElement>(null);
  const midpageAnchorRef = useRef<HTMLButtonElement>(null);

  const toggle = (key: keyof WindowVisibility) => {
    setVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  useEffect(() => {
    (window as unknown as Record<string, unknown>)['__overlayWindowTestHelpers'] = {
      toggle: (key: string) => toggle(key as keyof WindowVisibility),
      getVisibility: () => visibility,
      getClosedWindows: () => closedWindows,
    };
    return () => {
      delete (window as unknown as Record<string, unknown>)['__overlayWindowTestHelpers'];
    };
  }, [visibility, closedWindows]);

  const onHide = (windowName: string) => {
    setVisibility((prev) => ({ ...prev, [windowName]: false }));
    setClosedWindows((prev) => [...prev, windowName]);
  };

  return (
    <Page title="OverlayWindow Test">
      <Title>OverlayWindow Component Test</Title>

      <Section>
        <Subtitle>Controls</Subtitle>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          <AnchorButton data-testid="toggle-fixed-center" onClick={() => toggle('fixedCenter')}>
            Fixed Center {visibility.fixedCenter ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-fixed-bottom-right" onClick={() => toggle('fixedBottomRight')}>
            Fixed BottomRight {visibility.fixedBottomRight ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-fixed-custom" onClick={() => toggle('fixedCustom')}>
            Fixed Custom {visibility.fixedCustom ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-absolute-ref" ref={anchorRef} onClick={() => toggle('absoluteRef')}>
            Absolute + Ref {visibility.absoluteRef ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-absolute-custom" onClick={() => toggle('absoluteCustom')}>
            Absolute Custom {visibility.absoluteCustom ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-fixed-no-drag" onClick={() => toggle('fixedNoDrag')}>
            Fixed Locked {visibility.fixedNoDrag ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-headerless-menu" onClick={() => toggle('headerlessMenu')}>
            Headerless Menu {visibility.headerlessMenu ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-headerless-no-close" onClick={() => toggle('headerlessNoClose')}>
            Headerless No Close {visibility.headerlessNoClose ? '(ON)' : '(OFF)'}
          </AnchorButton>
          <AnchorButton data-testid="toggle-headerless-on-close" onClick={() => toggle('headerlessOnClose')}>
            Headerless onClose {visibility.headerlessOnClose ? '(ON)' : '(OFF)'}
          </AnchorButton>
        </div>
      </Section>

      <StatusBar data-testid="closed-windows">Closed: {closedWindows.join(', ') || 'none'}</StatusBar>

      <Spacer $height={400} data-label="scroll zone — 400px" />

      <ScrollMarker data-testid="scroll-marker-1">Scroll marker 1 — ~500px from top</ScrollMarker>

      <Spacer $height={400} data-label="scroll zone — 400px" />

      <Section>
        <Subtitle>Mid-page Anchor</Subtitle>
        <AnchorButton ref={midpageAnchorRef} data-testid="midpage-anchor" onClick={() => toggle('absoluteMidpage')}>
          Anchor button at ~950px {visibility.absoluteMidpage ? '(ON)' : '(OFF)'}
        </AnchorButton>
      </Section>

      <Spacer $height={600} data-label="scroll zone — 600px" />

      <ScrollMarker data-testid="scroll-marker-2">Scroll marker 2 — ~1600px from top</ScrollMarker>

      <Spacer $height={600} data-label="scroll zone — 600px" />

      <ScrollMarker data-testid="scroll-marker-3">Scroll marker 3 — ~2250px from top</ScrollMarker>

      <Spacer $height={400} data-label="scroll zone — 400px" />

      <ScrollMarker data-testid="scroll-marker-bottom">Bottom of page — ~2700px from top</ScrollMarker>

      {/* Fixed + Center: stays centered in viewport regardless of scroll */}
      <OverlayWindow
        show={visibility.fixedCenter}
        positioning={PositionType.Fixed}
        placement={Placement.Center}
        title="Fixed Center"
        width={300}
        height={200}
        onHide={() => onHide('fixedCenter')}
        className="test-window-fixed-center"
      >
        <WindowContent data-testid="window-fixed-center">
          Fixed positioning, centered in viewport. Stays in place when scrolling.
        </WindowContent>
      </OverlayWindow>

      {/* Fixed + BottomRight: stays in bottom-right of viewport */}
      <OverlayWindow
        show={visibility.fixedBottomRight}
        positioning={PositionType.Fixed}
        placement={Placement.BottomRight}
        title="Fixed BottomRight"
        width={280}
        height={180}
        onHide={() => onHide('fixedBottomRight')}
        className="test-window-fixed-bottom-right"
      >
        <WindowContent data-testid="window-fixed-bottom-right">
          Fixed positioning, bottom-right corner. Stays in place when scrolling.
        </WindowContent>
      </OverlayWindow>

      {/* Fixed + Custom: stays at explicit viewport coordinates */}
      <OverlayWindow
        show={visibility.fixedCustom}
        positioning={PositionType.Fixed}
        placement={Placement.Custom}
        customPosition={{ top: 100, left: 150 }}
        title="Fixed Custom Position"
        width={320}
        height={220}
        onHide={() => onHide('fixedCustom')}
        className="test-window-fixed-custom"
      >
        <WindowContent data-testid="window-fixed-custom">
          Fixed positioning at explicit viewport coordinates (top: 100, left: 150). Stays in place when scrolling.
        </WindowContent>
      </OverlayWindow>

      {/* Absolute + parentRef: positioned relative to the top controls button, scrolls with page */}
      <OverlayWindow
        show={visibility.absoluteRef}
        positioning={PositionType.Absolute}
        placement={Placement.Bottom}
        parentRef={anchorRef}
        title="Absolute + Ref"
        width={300}
        height={200}
        onHide={() => onHide('absoluteRef')}
        className="test-window-absolute-ref"
      >
        <WindowContent data-testid="window-absolute-ref">
          Absolute positioning relative to the &quot;Absolute + Ref&quot; button. Scrolls with the page.
        </WindowContent>
      </OverlayWindow>

      {/* Absolute + Custom: positioned at explicit document coordinates, scrolls with page */}
      <OverlayWindow
        show={visibility.absoluteCustom}
        positioning={PositionType.Absolute}
        placement={Placement.Custom}
        customPosition={{ top: 1600, left: 100 }}
        title="Absolute Custom Position"
        width={300}
        height={200}
        onHide={() => onHide('absoluteCustom')}
        className="test-window-absolute-custom"
      >
        <WindowContent data-testid="window-absolute-custom">
          Absolute positioning at document coordinates (top: 1600, left: 100). Scrolls with the page — scroll down to find it.
        </WindowContent>
      </OverlayWindow>

      {/* Absolute + parentRef (midpage): positioned relative to the mid-page anchor, scrolls with page */}
      <OverlayWindow
        show={visibility.absoluteMidpage}
        positioning={PositionType.Absolute}
        placement={Placement.Bottom}
        parentRef={midpageAnchorRef}
        title="Absolute Midpage Anchor"
        width={300}
        height={200}
        onHide={() => onHide('absoluteMidpage')}
        className="test-window-absolute-midpage"
      >
        <WindowContent data-testid="window-absolute-midpage">
          Absolute positioning relative to the mid-page anchor button. Scrolls with the page.
        </WindowContent>
      </OverlayWindow>

      {/* Fixed + locked: cannot be dragged or resized */}
      <OverlayWindow
        show={visibility.fixedNoDrag}
        positioning={PositionType.Fixed}
        placement={Placement.TopLeft}
        locked={true}
        title="Fixed Locked"
        width={260}
        height={160}
        onHide={() => onHide('fixedNoDrag')}
        className="test-window-fixed-no-drag"
      >
        <WindowContent data-testid="window-fixed-no-drag">Fixed positioning, top-left, locked (no drag or resize).</WindowContent>
      </OverlayWindow>

      {/* Headerless with onClose: dismisses on click-outside or Escape */}
      <OverlayWindow
        show={visibility.headerlessMenu}
        positioning={PositionType.Fixed}
        placement={Placement.Custom}
        customPosition={{ top: 200, left: 300 }}
        width={220}
        height={180}
        onHide={() => onHide('headerlessMenu')}
        onClose={() => onHide('headerlessMenu')}
        className="test-window-headerless-menu"
      >
        <WindowContent data-testid="window-headerless-menu">
          Headerless menu with onClose. Click outside or press Escape to dismiss.
        </WindowContent>
      </OverlayWindow>

      {/* Headerless without onHide or onClose: no close button, no dismiss, parent controls visibility */}
      <OverlayWindow
        show={visibility.headerlessNoClose}
        positioning={PositionType.Fixed}
        placement={Placement.Custom}
        customPosition={{ top: 200, left: 550 }}
        width={220}
        height={180}
        className="test-window-headerless-no-close"
      >
        <WindowContent data-testid="window-headerless-no-close">
          Headerless window without close button. Parent controls visibility.
        </WindowContent>
      </OverlayWindow>
      {/* Headerless with onClose only: no close button, dismisses on click-outside or Escape */}
      <OverlayWindow
        show={visibility.headerlessOnClose}
        positioning={PositionType.Fixed}
        placement={Placement.Custom}
        customPosition={{ top: 200, left: 550 }}
        width={220}
        height={180}
        onClose={() => onHide('headerlessOnClose')}
        className="test-window-headerless-on-close"
      >
        <WindowContent data-testid="window-headerless-on-close">
          Headerless window with onClose only. Click outside or press Escape to dismiss.
        </WindowContent>
      </OverlayWindow>
    </Page>
  );
};

export default OverlayWindowTest;
