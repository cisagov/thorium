import { ErrorBoundary } from 'react-error-boundary';
import { Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

import Page from '@components/pages/Page';
import Title from '@components/shared/titles/Title';
import Subtitle from '@components/shared/titles/Subtitle';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import RenderErrorAlert from '@components/shared/alerts/RenderErrorAlert';

// Throws during render to exercise an ErrorBoundary (driven by render-error.spec.ts)
const Thrower = () => {
  throw new Error('Intentional render error for verification');
};

const Section = styled.div`
  margin-bottom: 2rem;
`;

const SectionLabel = styled.h5`
  color: var(--thorium-text-muted, #999);
  margin-bottom: 0.75rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const AlertBannerTest = () => {
  // Query params drive render-error scenarios for render-error.spec.ts
  const params = new URLSearchParams(window.location.search);
  if (params.get('crash') === '1') {
    // page=true: error propagates to the top-level ErrorBoundary in Thorium.tsx
    return <Thrower />;
  }
  if (params.get('inline') === '1') {
    // page=false: a nested boundary renders the inline alert (as AssociationGraph/Tree do)
    return (
      <Page title="Inline Render Error Test">
        <div data-testid="inline-marker">marker above inline boundary</div>
        <ErrorBoundary fallback={<RenderErrorAlert page={false} />}>
          <Thrower />
        </ErrorBoundary>
      </Page>
    );
  }
  return (
    <Page title="AlertBanner Test">
      <Title>AlertBanner Component Test</Title>

      <Section>
        <Subtitle>Severity Variants</Subtitle>
        <SectionLabel>Default (Error)</SectionLabel>
        <AlertBanner>Something went wrong while processing your request.</AlertBanner>

        <SectionLabel className="mt-3">Warning</SectionLabel>
        <AlertBanner severity={Severity.Warning}>Your session will expire in 5 minutes.</AlertBanner>

        <SectionLabel className="mt-3">Info</SectionLabel>
        <AlertBanner severity={Severity.Info}>Results are still being processed. Check back shortly.</AlertBanner>

        <SectionLabel className="mt-3">Success</SectionLabel>
        <AlertBanner severity={Severity.Success}>File uploaded successfully!</AlertBanner>
      </Section>

      <Section>
        <Subtitle>Dismissible</Subtitle>
        <Row className="g-3">
          <Col md={6}>
            <SectionLabel>Dismissible Error</SectionLabel>
            <AlertBanner dismissible>This error can be dismissed.</AlertBanner>
          </Col>
          <Col md={6}>
            <SectionLabel>Dismissible Success</SectionLabel>
            <AlertBanner severity={Severity.Success} dismissible>
              Operation complete — dismiss when ready.
            </AlertBanner>
          </Col>
        </Row>
      </Section>

      <Section>
        <Subtitle>With className</Subtitle>
        <SectionLabel>className=&quot;m-4&quot;</SectionLabel>
        <AlertBanner severity={Severity.Warning} className="m-4">
          This banner has extra margin via className.
        </AlertBanner>

        <SectionLabel>className=&quot;mt-1 mb-0&quot;</SectionLabel>
        <AlertBanner severity={Severity.Info} className="mt-1 mb-0">
          Tight vertical spacing.
        </AlertBanner>
      </Section>

      <Section>
        <Subtitle>Rich Children</Subtitle>
        <SectionLabel>With inline code</SectionLabel>
        <AlertBanner>
          Failed to parse field <code>metadata.sha256</code> — expected a hex string.
        </AlertBanner>

        <SectionLabel className="mt-3">With h3 heading</SectionLabel>
        <AlertBanner severity={Severity.Info}>
          <h3>No Tool Results Available</h3>
        </AlertBanner>

        <SectionLabel className="mt-3">With pre-formatted text</SectionLabel>
        <AlertBanner>
          <pre>Error: ENOENT: no such file or directory, open &apos;/tmp/upload_cache/abc123&apos;</pre>
        </AlertBanner>

        <SectionLabel className="mt-3">With links and mixed content</SectionLabel>
        <AlertBanner severity={Severity.Success}>
          Reaction deleted successfully! Return to sample&nbsp;
          <a href="#" style={{ color: 'inherit', textDecoration: 'underline' }}>
            e3b0c44298fc1c149afbf4c8996fb924
          </a>
        </AlertBanner>
      </Section>

      <Section>
        <Subtitle>Long Content</Subtitle>
        <AlertBanner className="word-break-all">
          {`ErrorResponseStatus: PermissionDenied, upstream_error="user 'readonly_user' does not have permission 'admin:write' on resource 'groups/engineering-team/settings/notification_channels' required by endpoint PUT /api/v1/groups/{group}/settings/notification_channels"`}
        </AlertBanner>
      </Section>

      <Section>
        <Subtitle>Multiple Stacked</Subtitle>
        <AlertBanner>First error: connection timed out</AlertBanner>
        <AlertBanner className="mt-2">Second error: retry limit exceeded</AlertBanner>
        <AlertBanner className="mt-2" severity={Severity.Warning}>
          Warning: partial results returned
        </AlertBanner>
      </Section>
    </Page>
  );
};

export default AlertBannerTest;
