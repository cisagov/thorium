import React, { useState } from 'react';
import { Button, Modal } from 'react-bootstrap';
import { useNavigate, useSearchParams } from 'react-router';

// project imports
import { getBrowsingPathByEntity } from '@components/entities/browsing/EntityBrowsingRoutes';
import { buildTagBrowseHref, getTagBadgeText, getTagColorClass } from '@components/tags/utilities';
import { OverlayTipBottom } from '@components/shared/overlay/tips';
import { Entities, entityLabel } from '@models/entities';
import { TagUpperKeyEnum } from '@models/tags';

// spec: ./tags.spec.md

interface TagBadgeProps {
  tag: string;
  value: string;
  condensed: boolean;
  /** Which onclick behavior to render: 'scroll' (jump to results), 'docs' (mitre links), or 'link' (browse by tag). */
  action: string;
  resource?: Entities;
}

/** Truncate badge text to a fixed width with an ellipsis for the compact ("short-tag") badge variant. */
export const truncateBadgeText = (text: string): string => (text.length > 30 ? `${text.substring(0, 30)}...` : text);

/**
 * Render the paired full-width + compact badge divs shared by every TagBadge branch. The full-width
 * div (`tags-hide`) and the compact div (`short-tag`) are swapped by responsive CSS; the compact one
 * always shows the truncated text. `clickable` toggles the shared clickable styling/onClick.
 */
const BadgePair: React.FC<{ badgeClass: string; text: string; clickable: boolean; onClick?: () => void }> = ({
  badgeClass,
  text,
  clickable,
  onClick,
}) => {
  const clickableClass = clickable ? ' clickable' : '';
  return (
    <>
      <div className={`${badgeClass} ms-1 mb-1 tag-item${clickableClass} tags-hide`} onClick={onClick}>
        {text}
      </div>
      <div className={`${badgeClass} ms-1 mb-1 tag-item${clickableClass} short-tag`} onClick={onClick}>
        {truncateBadgeText(text)}
      </div>
    </>
  );
};

const TagBadge: React.FC<TagBadgeProps> = ({ tag, value, condensed, action, resource }) => {
  const [showRedirectModal, setShowRedirectModal] = useState(false);
  const badgeClass = getTagColorClass(tag, value);
  const [, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const tagText = getTagBadgeText(tag, value, condensed);
  const upperTag = tag.toUpperCase() as TagUpperKeyEnum;

  if (action == 'scroll') {
    const scrollToResult = (value: string) => {
      const element = document.getElementById(`results-tab-${value}`);
      if (element) {
        element.scrollIntoView();
      }
    };
    return (
      <OverlayTipBottom tip={`Click to jump to ${value} results`}>
        <BadgePair badgeClass={badgeClass} text={tagText} clickable onClick={() => scrollToResult(value)} />
      </OverlayTipBottom>
    );
    // link to external mitre docs for Att&ck tags
  } else if (action == 'docs' && upperTag == TagUpperKeyEnum.ATTACK) {
    const tactic = value.split(' ');
    const attackID = tactic.at(-1)?.split('.')[0];
    const attackSubID = tactic.at(-1)?.split('.').at(1);
    let redirectURL = '';
    if (attackSubID != undefined) {
      redirectURL = `https://attack.mitre.org/techniques/${attackID}/${attackSubID}/`;
    } else {
      redirectURL = `https://attack.mitre.org/techniques/${attackID}/`;
    }
    // on click function to redirect to external URL
    const redirectToExternal = () => {
      window.open(redirectURL, '_blank');
    };
    return (
      <>
        <Modal show={showRedirectModal} onHide={() => setShowRedirectModal(false)}>
          <Modal.Header closeButton>
            <h3>Navigate to an external site?</h3>
          </Modal.Header>
          <Modal.Body className="d-flex justify-content-center">
            <i>{redirectURL}</i>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              variant=""
              className="warning-btn"
              onClick={() => {
                redirectToExternal();
                setShowRedirectModal(false);
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
        <OverlayTipBottom tip={`Click to see mitre documentation on this technique: ${tagText}`}>
          <a className="no-decoration" onClick={() => setShowRedirectModal(true)}>
            <BadgePair badgeClass={badgeClass} text={tagText} clickable />
          </a>
        </OverlayTipBottom>
      </>
    );
    // link to external mitre docs for MBC tags
  } else if (action == 'docs' && upperTag == TagUpperKeyEnum.MBC) {
    const splitIndex = value.lastIndexOf(' ');
    const identifier = value.slice(splitIndex);
    const splitText = value.slice(0, splitIndex).split('::');
    const behavior = splitText[0].replaceAll(' ', '-').toLowerCase();
    const method = splitText[1].replaceAll(' ', '-').toLowerCase();
    let redirectURL = '';
    if (!identifier.includes('C')) {
      redirectURL = `https://github.com/MBCProject/mbc-markdown/tree/v3.0/${behavior}/${method}.md`;
    } else {
      redirectURL = `https://github.com/MBCProject/mbc-markdown/tree/v3.0/micro-behaviors/${behavior}/${method}.md`;
    }
    // on click function to redirect to external URL
    const redirectToExternal = () => {
      window.open(redirectURL, '_blank');
    };

    return (
      <>
        <Modal show={showRedirectModal} onHide={() => setShowRedirectModal(false)}>
          <Modal.Header closeButton>
            <h3>Navigate to an external site?</h3>
          </Modal.Header>
          <Modal.Body className="d-flex justify-content-center">
            <i>{redirectURL}</i>
          </Modal.Body>
          <Modal.Footer className="d-flex justify-content-center">
            <Button
              variant=""
              className="warning-btn"
              onClick={() => {
                redirectToExternal();
                setShowRedirectModal(false);
              }}
            >
              Confirm
            </Button>
          </Modal.Footer>
        </Modal>
        <OverlayTipBottom tip={`Click to see mitre documentation on this behavior: ${tagText}`}>
          <a className="no-decoration" onClick={() => setShowRedirectModal(true)}>
            <BadgePair badgeClass={badgeClass} text={tagText} clickable />
          </a>
        </OverlayTipBottom>
      </>
    );
  } else if (action == 'link') {
    // resolve the resource's browse route once; both the append-in-place check and the href need it
    const base = resource ? getBrowsingPathByEntity(resource) : undefined;
    const href = resource ? buildTagBrowseHref(resource, tag, value) : undefined;
    // no resource or no browse route → render a plain, non-clickable badge rather than a dead anchor
    if (!resource || !base || !href) {
      return (
        <div>
          <BadgePair badgeClass={badgeClass} text={tagText} clickable={false} />
        </div>
      );
    }
    const onClick = (e: React.MouseEvent) => {
      // let the browser handle modified clicks (new tab, etc.) via the real href
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
      e.preventDefault();
      // segment-aware match so '/files' does not also match '/filesystems' (a plain startsWith would)
      const onBrowsePage = window.location.pathname === base || window.location.pathname.startsWith(`${base}/`);
      // already on this resource's browse page → append the tag to the current filters (no reload);
      // otherwise SPA-navigate to the browse page pre-filtered by this tag
      if (onBrowsePage) {
        const query = new URLSearchParams(window.location.search);
        query.append(`tags[${tag}]`, value);
        setSearchParams(query, { replace: true });
      } else {
        void navigate(href);
      }
    };
    return (
      <OverlayTipBottom tip={`Click to browse ${entityLabel(resource)}s with tag: ${tagText}`}>
        <a className="no-decoration" href={href} onClick={onClick}>
          <BadgePair badgeClass={badgeClass} text={tagText} clickable />
        </a>
      </OverlayTipBottom>
    );
  } else {
    return (
      <div>
        <BadgePair badgeClass={badgeClass} text={tagText} clickable={false} />
      </div>
    );
  }
};

export default TagBadge;
