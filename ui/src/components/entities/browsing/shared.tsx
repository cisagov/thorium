import React, { useEffect, useState } from 'react';
import { Card, Col, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { scaling } from '@styles';
import { fetchEntityImage } from '@thorpi/entities';

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
  cursor: pointer;
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

export const EntitySecondary = styled(Col)`
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

const InlineIcon = styled.img`
  height: 1.2em;
  width: 1.2em;
  object-fit: contain;
  vertical-align: middle;
  margin-right: 10px;
  border-radius: 2px;
  flex-shrink: 0;
`;

const InlineSvgIcon = styled.div<{ $src: string }>`
  height: 1.2em;
  width: 1.2em;
  vertical-align: middle;
  margin-right: 10px;
  flex-shrink: 0;
  display: inline-block;
  background-color: currentColor;
  mask-image: url(${(props) => props.$src});
  mask-size: contain;
  mask-repeat: no-repeat;
  mask-position: center;
  -webkit-mask-image: url(${(props) => props.$src});
  -webkit-mask-size: contain;
  -webkit-mask-repeat: no-repeat;
  -webkit-mask-position: center;
`;

const NameWrapper = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
`;

interface EntityNameWithIconProps {
  entityId: string;
  hasImage: boolean;
  children: React.ReactNode;
}

export const EntityNameWithIcon: React.FC<EntityNameWithIconProps> = ({ entityId, hasImage, children }) => {
  const [iconUrl, setIconUrl] = useState<string | null>(null);
  const [isSvg, setIsSvg] = useState(false);

  useEffect(() => {
    if (hasImage) {
      void fetchEntityImage(entityId).then((img) => {
        if (img) {
          setIconUrl(img.url);
          setIsSvg(img.isSvg);
        }
      });
    }
  }, [entityId, hasImage]);

  return (
    <NameWrapper>
      {iconUrl && (isSvg ? <InlineSvgIcon $src={iconUrl} /> : <InlineIcon src={iconUrl} alt="" />)}
      {children}
    </NameWrapper>
  );
};
