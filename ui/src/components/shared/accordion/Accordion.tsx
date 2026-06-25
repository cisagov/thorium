import { createContext, useContext } from 'react';
import type { FC, ReactNode } from 'react';
import { FaChevronDown } from 'react-icons/fa';
import styled from 'styled-components';

// A themed, controlled accordion built with styled-components to replace the legacy react-bootstrap
// Accordion. The open set is owned by the parent (via `activeKey`/`onSelect`) so callers can expand
// items programmatically. `alwaysOpen` allows multiple items open at once; otherwise opening one
// closes the others (mirroring react-bootstrap's behavior).

interface AccordionContextValue {
  activeKeys: string[];
  toggle: (key: string) => void;
}

const AccordionContext = createContext<AccordionContextValue | undefined>(undefined);

function useAccordion(): AccordionContextValue {
  const ctx = useContext(AccordionContext);
  if (!ctx) throw new Error('Accordion subcomponents must be used within an <Accordion>');
  return ctx;
}

interface ItemContextValue {
  eventKey: string;
  isOpen: boolean;
}

const ItemContext = createContext<ItemContextValue | undefined>(undefined);

function useAccordionItem(): ItemContextValue {
  const ctx = useContext(ItemContext);
  if (!ctx) throw new Error('AccordionHeader/AccordionBody must be used within an <AccordionItem>');
  return ctx;
}

const AccordionContainer = styled.div`
  display: flex;
  flex-direction: column;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  overflow: hidden;

  /* collapse adjacent borders so items read as one stacked group */
  & > * + * {
    border-top: 1px solid var(--thorium-panel-border);
  }
`;

const HeaderRow = styled.div<{ $open: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 0.25rem 1.25rem;
  background-color: ${({ $open }) => ($open ? 'var(--thorium-secondary-panel-bg)' : 'var(--thorium-panel-bg)')};
  color: var(--thorium-secondary-text);
  cursor: pointer;
  user-select: none;

  &:hover {
    background-color: var(--thorium-highlight-panel-bg);
  }

  &:focus-visible {
    outline: none;
    box-shadow:
      inset 0 0 1px var(--thorium-panel-border),
      0 0 8px var(--thorium-highlight-panel-border);
  }
`;

const HeaderContent = styled.div`
  display: flex;
  flex: 1;
  align-items: center;
  min-width: 0;
`;

const Chevron = styled(FaChevronDown)<{ $open: boolean }>`
  flex: 0 0 auto;
  color: var(--thorium-secondary-text);
  transition: transform 0.2s ease;
  transform: rotate(${({ $open }) => ($open ? '180deg' : '0deg')});
`;

const ItemWrapper = styled.div`
  display: flex;
  flex-direction: column;
`;

const BodyWrapper = styled.div`
  padding: 1rem 1.25rem;
  background-color: var(--thorium-panel-bg);
  color: var(--thorium-text);
`;

interface AccordionProps {
  /// The currently-open item keys (controlled).
  activeKey: string[];
  /// Called with the next set of open keys whenever an item is toggled.
  onSelect: (keys: string[]) => void;
  /// Allow multiple items to be open simultaneously.
  alwaysOpen?: boolean;
  children: ReactNode;
}

/// Controlled accordion root. Owns no open state itself — the parent supplies `activeKey`.
export const Accordion: FC<AccordionProps> = ({ activeKey, onSelect, alwaysOpen = false, children }) => {
  const toggle = (key: string) => {
    const isOpen = activeKey.includes(key);
    if (alwaysOpen) {
      onSelect(isOpen ? activeKey.filter((k) => k !== key) : [...activeKey, key]);
    } else {
      onSelect(isOpen ? [] : [key]);
    }
  };

  return (
    <AccordionContext.Provider value={{ activeKeys: activeKey, toggle }}>
      <AccordionContainer data-testid="accordion">{children}</AccordionContainer>
    </AccordionContext.Provider>
  );
};

interface AccordionItemProps {
  eventKey: string;
  children: ReactNode;
}

/// A single accordion item; exposes its key and open state to its header/body.
export const AccordionItem: FC<AccordionItemProps> = ({ eventKey, children }) => {
  const { activeKeys } = useAccordion();
  const isOpen = activeKeys.includes(eventKey);
  return (
    <ItemContext.Provider value={{ eventKey, isOpen }}>
      <ItemWrapper data-testid="accordion-item" id={eventKey} className="scrollable-item">
        {children}
      </ItemWrapper>
    </ItemContext.Provider>
  );
};

interface AccordionHeaderProps {
  children: ReactNode;
}

/// The clickable header that toggles its item. Uses a div (not a button) so it can safely contain
/// action buttons without nesting interactive elements.
export const AccordionHeader: FC<AccordionHeaderProps> = ({ children }) => {
  const { toggle } = useAccordion();
  const { eventKey, isOpen } = useAccordionItem();
  const onToggle = () => toggle(eventKey);
  return (
    <HeaderRow
      $open={isOpen}
      data-testid="accordion-header"
      role="button"
      tabIndex={0}
      aria-expanded={isOpen}
      onClick={onToggle}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onToggle();
        }
      }}
    >
      <HeaderContent>{children}</HeaderContent>
      <Chevron $open={isOpen} />
    </HeaderRow>
  );
};

interface AccordionBodyProps {
  children: ReactNode;
}

/// The collapsible body; rendered only while its item is open.
export const AccordionBody: FC<AccordionBodyProps> = ({ children }) => {
  const { isOpen } = useAccordionItem();
  if (!isOpen) return null;
  return <BodyWrapper data-testid="accordion-body">{children}</BodyWrapper>;
};
