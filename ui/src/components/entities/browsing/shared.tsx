import { Card, Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { scaling } from '@styles';

export const BrowsingCard = styled(Card)`
  margin: 0.1em 0em 0em 0.1em;
  min-height: 3em;
  color: var(--thorium-text);
  background-color: var(--thorium-panel-bg);
`;

export const BrowsingContents = styled(Card.Body)`
  flex-wrap: wrap;
  flex: 1 1 auto;
  padding: var(--bs-card-spacer-y) var(--bs-card-spacer-x);
  color: var(--bs-card-color);
`;

export const LinkFields = styled(Row)`
  display: flex;
  flex-wrap: wrap;
  color: var(--thorium-text);
  background-color: var(--thorium-panel-bg);

  &:hover {
    color: var(--thorium-text);
    background-color: var(--thorium-highlight-panel-bg);
    box-shadow:
      inset 0 0 1px var(--thorium-panel-border),
      0 0 4px var(--thorium-highlight-panel-border) !important;
    // this makes sure the box shadow isn't hidden behind the card above
    z-index: 1000;
  }
`;

export const EntityName = styled(Col)`
  white-space: pre-wrap;
  text-align: center;
  flex-wrap: wrap;
  word-break: break-all;
  min-width: 400px;
  color: var(--thorium-text);
`;

export const EntityOrigin = styled(Col)`
  min-width: 300px;
  text-align: center;
  color: var(--thorium-text);
  @media (max-width: ${scaling.lg}) {
    display: none !important;
  }
`;

export const EntitySubmitters = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xxl}) {
    display: none !important;
  }
`;

export const EntityGroups = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 200px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xl}) {
    display: none !important;
  }
`;
