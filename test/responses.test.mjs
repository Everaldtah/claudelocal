import test from 'node:test';
import assert from 'node:assert/strict';
import { toResponsesRequest, responsesTranslator, toAnthropicMessage } from '../src/responses.mjs';
const frame = e => `data: ${JSON.stringify(e)}\r\n\r\n`;
const completed = (output = [], usage = {}) => ({ type: 'response.completed', response: { id: 'resp_test', status: 'completed', output, usage } });
const message = text => ({ type: 'message', content: [{ type: 'output_text', text }] });

test('history preserves call IDs, results, images; unsupported request fields stay absent', () => {
  const request = toResponsesRequest({ system: 'System instruction', stream: false, max_tokens: 100, temperature: 1,
    tools: [{ name: 'Read', input_schema: { type: 'object', properties: {} } }],
    tool_choice: { type: 'tool', name: 'Read', disable_parallel_tool_use: true },
    messages: [
      { role: 'user', content: 'Read the image' },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'call_123', name: 'Read', input: { path: 'image.png' } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call_123', content: [
        { type: 'text', text: 'Image loaded' }, { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAA' } },
      ] }] },
    ],
  }, { model: 'gpt-test', extra: { store: true, model: 'wrong', reasoning_effort: 'low' } });
  assert.equal(request.instructions, 'System instruction');
  assert.equal(request.input[1].call_id, 'call_123');
  assert.equal(request.input[1].arguments, '{"path":"image.png"}');
  assert.equal(request.input[2].call_id, 'call_123');
  assert.equal(request.input[2].output, 'Image loaded');
  assert.equal(request.input[3].content[0].image_url, 'data:image/png;base64,AAAA');
  assert.equal(request.tools[0].name, 'Read');
  assert.deepEqual(request.tool_choice, { type: 'function', name: 'Read' });
  assert.equal(request.parallel_tool_calls, false);
  assert.equal(request.stream, true);
  assert.equal(request.store, false);
  assert.equal(request.model, 'gpt-test');
  assert.equal(request.max_tokens, undefined);
  assert.equal(request.temperature, undefined);
});

test('UTF-8 and CRLF may be split at every byte; cached input is not double counted', () => {
  let out = '';
  const translator = responsesTranslator({ model: 'test', onChunk: c => out += c });
  const input = frame({ type: 'response.created', response: { id: 'resp_test' } }) +
    frame({ type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: 'café 🚀' }) +
    frame({ type: 'response.output_text.done', output_index: 0, content_index: 0, text: 'café 🚀' }) +
    frame(completed([message('café 🚀')], { input_tokens: 100, input_tokens_details: { cached_tokens: 80 }, output_tokens: 5 }));
  for (const byte of Buffer.from(input)) translator.push(Buffer.from([byte]));
  const result = translator.end();
  assert.equal(result.content[0].text, 'café 🚀');
  assert.equal(result.usage.input_tokens, 20);
  assert.equal(result.usage.cache_read_input_tokens, 80);
  assert.equal((out.match(/event: message_stop/g) || []).length, 1);
  assert.equal((out.match(/café 🚀/g) || []).length, 1);
});

test('parallel tool calls use call_id and independently assemble arguments', () => {
  let out = '';
  const translator = responsesTranslator({ onChunk: c => out += c });
  const items = [0, 1].map(i => ({ type: 'function_call', id: `fc_${i}`, call_id: `call_${i}`, name: 'Read', arguments: '{"n":' + i + '}' }));
  for (const [i, item] of items.entries()) translator.push(frame({ type: 'response.output_item.added', output_index: i, item: { ...item, arguments: '' } }));
  for (const [i, item] of items.entries()) {
    translator.push(frame({ type: 'response.function_call_arguments.delta', output_index: i, delta: item.arguments }));
    translator.push(frame({ type: 'response.output_item.done', output_index: i, item }));
  }
  translator.push(frame(completed(items)));
  const result = translator.end();
  assert.equal(result.stop_reason, 'tool_use');
  assert.deepEqual(result.content.map(c => [c.id, c.input.n]), [['call_0', 0], ['call_1', 1]]);
  assert.equal((out.match(/event: content_block_start/g) || []).length, 2);
  assert.equal((out.match(/event: content_block_stop/g) || []).length, 2);
});

test('missing terminal event and provider failure never signal success', () => {
  let out = '';
  const translator = responsesTranslator({ onChunk: c => out += c });
  translator.push(frame({ type: 'response.output_text.delta', output_index: 0, delta: 'partial' }));
  assert.throws(() => translator.end(), /before response.completed/);
  assert.ok(!out.includes('event: message_stop'));
  assert.throws(() => translator.push(frame({ type: 'response.failed', response: { error: { message: 'quota reached' } } })), /quota reached/);
});

test('invalid tool arguments are rejected rather than changed to an empty object', () => {
  assert.throws(() => toAnthropicMessage({ output: [{ type: 'function_call', arguments: '{broken' }] }), SyntaxError);
});

test('max output truncation is distinct from failure', () => {
  const translator = responsesTranslator();
  translator.push(frame({ type: 'response.incomplete', response: {
    id: 'resp_test', status: 'incomplete', output: [message('partial')], incomplete_details: { reason: 'max_output_tokens' },
  } }));
  assert.equal(translator.end().stop_reason, 'max_tokens');
  const failure = responsesTranslator();
  assert.throws(() => failure.push(frame({ type: 'response.incomplete', response: { incomplete_details: { reason: 'other' } } })), /incomplete/);
});
