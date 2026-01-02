# Tower UI Components Library

A comprehensive, accessible Vue 3 component library designed for the Tower ecosystem with full TypeScript support and Claude Console theme integration.

## Components Overview

### TowerCard
Flexible card component with header/body/footer slots.

**Props:**
- `variant`: `'default' | 'elevated' | 'outlined'` (default: `'default'`)
- `noPadding`: `boolean` (default: `false`) - Remove default body padding
- `fullWidth`: `boolean` (default: `false`) - Make card take full width

**Slots:**
- `header` - Card header content
- `default` - Card body content
- `footer` - Card footer content

**Usage:**
```vue
<TowerCard variant="elevated" full-width>
  <template #header>
    <h2>Card Title</h2>
  </template>
  Card content goes here
  <template #footer>
    <button>Action</button>
  </template>
</TowerCard>
```

### TowerButton
Comprehensive button component with variants, loading states, and full accessibility.

**Props:**
- `variant`: `'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'ghost' | 'link'` (default: `'primary'`)
- `size`: `'xs' | 'sm' | 'md' | 'lg' | 'xl'` (default: `'md'`)
- `type`: `'button' | 'submit' | 'reset'` (default: `'button'`)
- `disabled`: `boolean` (default: `false`)
- `loading`: `boolean` (default: `false`) - Shows spinner
- `fullWidth`: `boolean` (default: `false`)
- `rounded`: `boolean` (default: `false`) - Fully rounded button
- `ariaLabel`: `string` - Accessibility label
- `ariaDescribedby`: `string` - Accessibility description

**Events:**
- `@click` - Button click event

**Slots:**
- `default` - Button text
- `icon` - Button icon (positioned before text)

**Usage:**
```vue
<TowerButton
  variant="primary"
  size="lg"
  :loading="isLoading"
  @click="handleSubmit"
>
  Submit Form
</TowerButton>
```

### TowerInput
Advanced input component with validation, error states, and accessibility features.

**Props:**
- `modelValue`: `string | number` - v-model value
- `type`: `'text' | 'email' | 'password' | 'number' | 'tel' | 'url' | 'search' | 'date' | 'time' | 'datetime-local'`
- `label`: `string` - Input label
- `placeholder`: `string` - Placeholder text
- `helperText`: `string` - Helper text below input
- `error`: `string` - Error message
- `size`: `'sm' | 'md' | 'lg'` (default: `'md'`)
- `variant`: `'default' | 'filled'` (default: `'default'`)
- `disabled`: `boolean` (default: `false`)
- `readonly`: `boolean` (default: `false`)
- `required`: `boolean` (default: `false`)
- `clearable`: `boolean` (default: `false`) - Show clear button
- `showCharacterCount`: `boolean` - Show character count
- `maxlength`: `string | number` - Maximum character length

**Events:**
- `@update:modelValue` - v-model update
- `@change` - Input change event
- `@blur` - Input blur event
- `@focus` - Input focus event
- `@clear` - Clear button clicked
- `@keydown` - Keydown event

**Slots:**
- `prefix` - Icon/content before input
- `suffix` - Icon/content after input

**Usage:**
```vue
<TowerInput
  v-model="email"
  type="email"
  label="Email Address"
  placeholder="Enter your email"
  helper-text="We'll never share your email"
  :error="emailError"
  required
  clearable
/>
```

### TowerModal
Accessible modal component with focus management and keyboard navigation.

**Props:**
- `open`: `boolean` - Modal visibility (v-model)
- `title`: `string` - Modal title
- `size`: `'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'full'` (default: `'md'`)
- `showCloseButton`: `boolean` (default: `true`)
- `closeOnOverlay`: `boolean` (default: `true`)
- `closeOnEscape`: `boolean` (default: `true`)
- `persistent`: `boolean` (default: `false`) - Cannot be closed

**Events:**
- `@update:open` - v-model update
- `@close` - Modal close event
- `@open` - Modal open event
- `@after-open` - After modal opens (transition complete)
- `@after-close` - After modal closes (transition complete)

**Slots:**
- `header` - Custom header content
- `default` - Modal body content
- `footer` - Modal footer content

**Usage:**
```vue
<TowerModal
  v-model:open="showModal"
  title="Confirm Action"
  size="sm"
  @close="handleClose"
>
  Are you sure you want to delete this item?
  <template #footer>
    <TowerButton @click="confirmDelete">Delete</TowerButton>
    <TowerButton variant="secondary" @click="showModal = false">Cancel</TowerButton>
  </template>
</TowerModal>
```

### TowerSelect
Accessible dropdown select component with keyboard navigation.

**Props:**
- `modelValue`: `string | number | object` - Selected value (v-model)
- `options`: `Array` - Array of options
- `label`: `string` - Select label
- `placeholder`: `string` (default: `'Select an option'`)
- `helperText`: `string` - Helper text
- `error`: `string` - Error message
- `disabled`: `boolean` (default: `false`)
- `required`: `boolean` (default: `false`)
- `noOptionsText`: `string` (default: `'No options available'`)
- `valueKey`: `string` (default: `'value'`) - Key for option value (object options)
- `labelKey`: `string` (default: `'label'`) - Key for option label (object options)
- `disabledKey`: `string` (default: `'disabled'`) - Key for disabled state (object options)

**Events:**
- `@update:modelValue` - v-model update
- `@change` - Selection change event
- `@open` - Dropdown opened
- `@close` - Dropdown closed

**Slots:**
- `option` - Custom option rendering

**Usage:**
```vue
<TowerSelect
  v-model="selectedUser"
  :options="users"
  label="Select User"
  value-key="id"
  label-key="name"
  disabled-key="inactive"
/>
```

### TowerTextarea
Multi-line text input with auto-resize and character counting.

**Props:**
- `modelValue`: `string` - Textarea value (v-model)
- `label`: `string` - Textarea label
- `placeholder`: `string` - Placeholder text
- `helperText`: `string` - Helper text
- `error`: `string` - Error message
- `rows`: `number | string` (default: `4`) - Visible rows
- `autoResize`: `boolean` (default: `false`) - Auto-resize based on content
- `resize`: `'none' | 'both' | 'horizontal' | 'vertical'` (default: `'vertical'`)
- `disabled`: `boolean` (default: `false`)
- `readonly`: `boolean` (default: `false`)
- `required`: `boolean` (default: `false`)
- `maxlength`: `string | number` - Maximum character length
- `showCharacterCount`: `boolean` - Show character count (auto if maxlength set)
- `showWordCount`: `boolean` (default: `false`) - Show word count
- `minHeight`: `string | number` - Minimum height for auto-resize
- `maxHeight`: `string | number` - Maximum height for auto-resize

**Events:**
- `@update:modelValue` - v-model update
- `@change` - Textarea change event
- `@blur` - Textarea blur event
- `@focus` - Textarea focus event
- `@keydown` - Keydown event

**Usage:**
```vue
<TowerTextarea
  v-model="description"
  label="Description"
  placeholder="Enter description..."
  auto-resize
  :maxlength="500"
  show-word-count
/>
```

## Accessibility Features

All components include comprehensive accessibility features:

- **Keyboard Navigation**: Full keyboard support (Tab, Enter, Escape, Arrow keys)
- **Screen Reader Support**: Proper ARIA labels, roles, and states
- **Focus Management**: Visual focus indicators and proper focus handling
- **High Contrast Support**: Increased border widths in high contrast mode
- **Reduced Motion**: Respects `prefers-reduced-motion` setting
- **Color Contrast**: WCAG 2.1 AA compliant color combinations

## Theme Integration

Components automatically integrate with the Claude Console theme using CSS custom properties:

```css
:root {
  --bg-primary: #1a1a1a;
  --bg-secondary: #2a2a2a;
  --text-primary: #e0e0e0;
  --accent-primary: #7aa2f7;
  /* ... more theme variables */
}
```

## Global Registration

Components are automatically registered globally in `main.js`:

```js
import {
  TowerCard,
  TowerButton,
  TowerInput,
  TowerModal,
  TowerSelect,
  TowerTextarea
} from './components'

app.component('TowerCard', TowerCard)
// ... other components
```

## TypeScript Support

All components are built with TypeScript and include full type definitions for props, events, and slots.

## Browser Support

- Modern browsers with ES2020+ support
- Vue 3.5+ required
- Tailwind CSS 3.4+ for styling utilities