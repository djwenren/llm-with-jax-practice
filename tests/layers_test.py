"""Tests for layers."""

import jax

jax.config.update("jax_num_cpu_devices", 8)

import jax.numpy as jnp  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position
import einops  # pylint: disable=wrong-import-position

from flax import nnx  # pylint: disable=wrong-import-position
from jax.sharding import PartitionSpec as P  # pylint: disable=wrong-import-position

from llm_with_jax_practice import layers  # pylint: disable=wrong-import-position
from llm_with_jax_practice import (
    sharding as _sharding,
)  # pylint: disable=wrong-import-position


def _make_cpu_mesh(shape, axes):
    """Create a mesh backed by simulated CPU devices.

    Using explicit CPU devices ensures the mesh works even when a GPU is
    present, because ``jax_num_cpu_devices`` always provides the requested
    number of virtual CPUs regardless of the accelerator backend.
    """
    return jax.make_mesh(shape, axes, devices=jax.devices("cpu"))


class TestLayers:
    """Tests for layers."""

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_linear(
        self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff
    ):
        """Test linear layer reference implementation."""
        w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
        linear = layers.Linear(
            in_features=d_model, out_features=d_ff, rngs=nnx.Rngs(jax.random.key(42))
        )
        linear.weight = jnp.array(w1_weight).transpose()

        call = (
            nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)
        )

        y = call(linear, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_linear")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_embedding(
        self, use_jit, numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model
    ):
        """Test embedding layer."""
        embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
        embedding = layers.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            rngs=nnx.Rngs(jax.random.key(42)),
        )
        embedding.weight = jnp.array(embedding_weight)

        call = (
            nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)
        )

        y = call(embedding, jnp.array(in_indices))
        numpy_snapshot.assert_match(y, test_name="test_embedding")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_rmsnorm(self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings):
        """Test RMSNorm layer."""
        state_dict, _ = ts_state_dict
        reference_weights = state_dict["layers.1.ln1.weight"]
        d_model = reference_weights.shape[0]
        rms_norm = layers.RMSNorm(d_model=d_model, eps=1e-5)
        rms_norm.weight = jnp.array(reference_weights)

        call = (
            nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)
        )

        y = call(rms_norm, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_rmsnorm")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_swiglu(
        self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff
    ):
        """Test SwiGLU layer."""
        w1_weight, w2_weight, w3_weight = [
            ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]
        ]
        swiglu = layers.SwiGLU(
            d_model=d_model, d_ff=d_ff, rngs=nnx.Rngs(jax.random.key(42))
        )
        swiglu.w1_projection.weight = jnp.array(w1_weight).transpose()
        swiglu.w3_projection.weight = jnp.array(w3_weight).transpose()
        swiglu.w2_projection.weight = jnp.array(w2_weight).transpose()

        call = (
            nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)
        )

        y = call(swiglu, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_swiglu")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_rope(
        self, use_jit, numpy_snapshot, in_embeddings, d_model, theta, n_queries, pos_ids
    ):
        """Test RoPE layer."""
        rope = layers.RoPE(theta=theta, d_k=d_model, max_seq_len=n_queries)

        call = (
            nnx.jit(lambda model, x, p: model(x, p))
            if use_jit
            else lambda model, x, p: model(x, p)
        )

        y = call(rope, jnp.array(in_embeddings), jnp.array(pos_ids))
        numpy_snapshot.assert_match(y, test_name="test_rope")

    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("attention_type", ["custom", "xla"])
    def test_multihead_self_attention(
        self,
        use_jit,
        attention_type,
        numpy_snapshot,
        in_embeddings,
        d_model,
        n_heads,
        ts_state_dict,
    ):
        """Test Multi-head self-attention layer."""
        d, _ = ts_state_dict
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
            d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
        ]
        multi_head_self_attention = layers.MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=n_heads,
            rngs=nnx.Rngs(jax.random.key(42)),
            attention_type=attention_type,
        )
        multi_head_self_attention.combined_in_projection.weight = jnp.concatenate(
            [
                jnp.array(q_proj_weight).transpose(),
                jnp.array(k_proj_weight).transpose(),
                jnp.array(v_proj_weight).transpose(),
            ],
            axis=-1,
        )
        multi_head_self_attention.out_projection.weight = jnp.array(
            o_proj_weight
        ).transpose()

        call = (
            nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)
        )

        y = call(multi_head_self_attention, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_multihead_self_attention")

    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("attention_type", ["custom", "xla"])
    def test_multihead_self_attention_with_rope(
        self,
        use_jit,
        attention_type,
        numpy_snapshot,
        in_embeddings,
        d_model,
        n_heads,
        ts_state_dict,
        n_keys,
        theta,
        pos_ids,
    ):
        """Test Multi-head self-attention layer with RoPE."""
        d, _ = ts_state_dict
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
            d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
        ]
        pos_ids = einops.rearrange(jnp.array(pos_ids), "seq -> 1 seq")
        rope = layers.RoPE(theta=theta, d_k=d_model // n_heads, max_seq_len=n_keys)
        multi_head_self_attention = layers.MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=n_heads,
            rngs=nnx.Rngs(jax.random.key(42)),
            attention_type=attention_type,
        )
        multi_head_self_attention.combined_in_projection.weight = jnp.concatenate(
            [
                jnp.array(q_proj_weight).transpose(),
                jnp.array(k_proj_weight).transpose(),
                jnp.array(v_proj_weight).transpose(),
            ],
            axis=-1,
        )
        multi_head_self_attention.out_projection.weight = jnp.array(
            o_proj_weight
        ).transpose()

        call = (
            nnx.jit(
                lambda model, x, rope, token_positions: model(
                    x, token_positions=token_positions, rope=rope
                )
            )
            if use_jit
            else lambda model, x, rope, token_positions: model(
                x, token_positions=token_positions, rope=rope
            )
        )

        y = call(
            multi_head_self_attention,
            jnp.array(in_embeddings),
            token_positions=jnp.array(pos_ids),
            rope=rope,
        )
        numpy_snapshot.assert_match(
            y, test_name="test_multihead_self_attention_with_rope"
        )

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_transformer_block(
        self,
        use_jit,
        numpy_snapshot,
        ts_state_dict,
        in_embeddings,
        d_model,
        n_heads,
        d_ff,
        n_keys,
        theta,
    ):
        """Test Transformer block."""
        block_weights = {
            k.replace("layers.0.", ""): v
            for k, v in ts_state_dict[0].items()
            if "layers.0." in k
        }
        rope = layers.RoPE(theta=theta, d_k=d_model // n_heads, max_seq_len=n_keys)
        transformer_block = layers.TransformerBlock(
            d_model=d_model,
            num_heads=n_heads,
            d_ff=d_ff,
            rngs=nnx.Rngs(jax.random.key(42)),
        )
        transformer_block.rms_norm_pre_attn.weight = nnx.Param(
            jnp.array(block_weights["ln1.weight"])
        )
        transformer_block.attn.combined_in_projection.weight = nnx.Param(
            jnp.concatenate(
                [
                    jnp.array(block_weights["attn.q_proj.weight"]).transpose(),
                    jnp.array(block_weights["attn.k_proj.weight"]).transpose(),
                    jnp.array(block_weights["attn.v_proj.weight"]).transpose(),
                ],
                axis=-1,
            )
        )
        transformer_block.attn.out_projection.weight = nnx.Param(
            jnp.array(block_weights["attn.output_proj.weight"]).transpose()
        )

        transformer_block.rms_norm_pre_ff.weight = nnx.Param(
            jnp.array(block_weights["ln2.weight"])
        )
        transformer_block.ffn.w1_projection.weight = nnx.Param(
            jnp.array(block_weights["ffn.w1.weight"]).transpose()
        )
        transformer_block.ffn.w3_projection.weight = nnx.Param(
            jnp.array(block_weights["ffn.w3.weight"]).transpose()
        )
        transformer_block.ffn.w2_projection.weight = nnx.Param(
            jnp.array(block_weights["ffn.w2.weight"]).transpose()
        )

        call = (
            nnx.jit(
                lambda model, x, rope, token_positions: model(
                    in_features=x,
                    token_positions=token_positions,
                    rope=rope,
                )
            )
            if use_jit
            else lambda model, x, rope, token_positions: model(
                in_features=x,
                token_positions=token_positions,
                rope=rope,
            )
        )

        y = call(
            transformer_block,
            jnp.array(in_embeddings),
            token_positions=jnp.arange(in_embeddings.shape[-2]),
            rope=rope,
        )
        numpy_snapshot.assert_match(y, test_name="test_transformer_block")


@pytest.mark.cpu_mesh
class TestLayersCpuMesh:
    """Sharding tests that use a simulated 8-device (4×2) CPU mesh.

    ``jax_num_cpu_devices`` is set to 8 at module import time, so these tests
    run correctly even on machines that have a GPU — JAX's CPU backend always
    provides the requested virtual devices.
    """

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_linear_sharding(
        self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff
    ):
        """Test linear layer with sharding."""
        w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            linear = layers.Linear(
                in_features=d_model,
                out_features=d_ff,
                rngs=nnx.Rngs(jax.random.key(42)),
                sharding=_sharding.LinearSharding(weight=P(None, "Y")),
            )
            call = (
                nnx.jit(lambda model, x: model(x))
                if use_jit
                else lambda model, x: model(x)
            )
            x = jax.device_put(jnp.array(in_embeddings), P("X", None))
            y = call(linear, x)
            assert y.sharding.spec == P("X", None, "Y")

            linear.weight = jax.device_put(
                jnp.array(w1_weight).transpose(), P(None, "Y")
            )
            y = call(linear, x)
            numpy_snapshot.assert_match(y, test_name="test_linear")
            assert y.sharding.spec == P("X", None, "Y")

    def test_linear_sharding_and_reduce_scatter(
        self, numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff
    ):
        """Test linear layer with sharding and reduce-scatter output."""
        w1_weight = ts_state_dict[0]["layers.0.ffn.w1.weight"]
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            linear = layers.Linear(
                in_features=d_model,
                out_features=d_ff,
                rngs=nnx.Rngs(jax.random.key(42)),
                sharding=_sharding.LinearSharding(
                    weight=P("Y", None), out=P("X", None, "Y")
                ),
            )

            @nnx.jit
            def call(model, x):
                return model(x)

            x = jax.device_put(jnp.array(in_embeddings), P("X", None, "Y"))
            y = call(linear, x)
            assert y.sharding.spec == P("X", None, "Y")

            linear.weight = jax.device_put(
                jnp.array(w1_weight).transpose(), P("Y", None)
            )
            y = call(linear, x)
            numpy_snapshot.assert_match(y, test_name="test_linear")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_embedding_sharding(
        self, use_jit, numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model
    ):
        """Test embedding layer with sharding."""
        embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            embedding = layers.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=d_model,
                rngs=nnx.Rngs(jax.random.key(42)),
                sharding=_sharding.EmbeddingSharding(
                    embedding_matrix=P(None, "Y"), out=P("X", None, "Y")
                ),
            )
            call = (
                nnx.jit(lambda model, x: model(x))
                if use_jit
                else lambda model, x: model(x)
            )
            x = jax.device_put(jnp.array(in_indices), P("X", None))
            y = call(embedding, x)
            assert y.sharding.spec == P("X", None, "Y")

            embedding.weight = jax.device_put(jnp.array(embedding_weight), P(None, "Y"))
            y = call(embedding, x)
            numpy_snapshot.assert_match(y, test_name="test_embedding")
            assert y.sharding.spec == P("X", None, "Y")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_rmsnorm_sharding(
        self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings
    ):
        """Test RMSNorm layer with sharding."""
        state_dict, _ = ts_state_dict
        reference_weights = state_dict["layers.1.ln1.weight"]
        d_model = reference_weights.shape[0]
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            rms_norm = layers.RMSNorm(
                d_model=d_model,
                eps=1e-5,
                sharding=_sharding.RMSNormSharding(
                    weight=P(
                        None,
                    )
                ),
            )
            call = (
                nnx.jit(lambda model, x: model(x))
                if use_jit
                else lambda model, x: model(x)
            )
            x = jax.device_put(jnp.array(in_embeddings), P("X", None))
            y = call(rms_norm, x)
            assert y.sharding.spec == P("X", None, None)

            rms_norm.weight = jax.device_put(jnp.array(reference_weights), P(None))
            y = call(rms_norm, x)
            numpy_snapshot.assert_match(y, test_name="test_rmsnorm")
            assert y.sharding.spec == P("X", None, None)

        call = (
            nnx.jit(lambda model, x: model(x)) if use_jit else lambda model, x: model(x)
        )

        y = call(rms_norm, jnp.array(in_embeddings))
        numpy_snapshot.assert_match(y, test_name="test_rmsnorm")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_swiglu_sharding(
        self, use_jit, numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff
    ):
        """Test SwiGLU layer with sharding."""
        w1_weight, w2_weight, w3_weight = [
            ts_state_dict[0][f"layers.0.ffn.{k}.weight"] for k in ["w1", "w2", "w3"]
        ]
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            swiglu = layers.SwiGLU(
                d_model=d_model,
                d_ff=d_ff,
                rngs=nnx.Rngs(jax.random.key(42)),
                sharding=_sharding.SwiGLUSharding(
                    up_projection=_sharding.LinearSharding(
                        weight=P("X", "Y"), out=P("X", None, "Y")
                    ),
                    down_projection=_sharding.LinearSharding(
                        weight=P("Y", "X"), out=P("X", None, "Y")
                    ),
                ),
            )

            call = (
                nnx.jit(lambda model, x: model(x))
                if use_jit
                else lambda model, x: model(x)
            )

            x = jax.device_put(jnp.array(in_embeddings), P("X", None, "Y"))
            y = call(swiglu, x)
            assert y.sharding.spec == P("X", None, "Y")

            swiglu.w1_projection.weight = jax.device_put(
                jnp.array(w1_weight).transpose(), P("X", "Y")
            )
            swiglu.w3_projection.weight = jax.device_put(
                jnp.array(w3_weight).transpose(), P("X", "Y")
            )
            swiglu.w2_projection.weight = jax.device_put(
                jnp.array(w2_weight).transpose(), P("Y", "X")
            )
            y = call(swiglu, x)
            numpy_snapshot.assert_match(y, test_name="test_swiglu")
            assert y.sharding.spec == P("X", None, "Y")

    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("attention_type", ["custom", "xla"])
    def test_multihead_self_attention_sharding(
        self,
        use_jit,
        attention_type,
        numpy_snapshot,
        ts_state_dict,
        in_embeddings,
        d_model,
        n_heads,
    ):
        """Test Multi-head self-attention layer with FSDP + TP sharding."""
        d, _ = ts_state_dict
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
            d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
        ]
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            multi_head_self_attention = layers.MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=n_heads,
                rngs=nnx.Rngs(jax.random.key(42)),
                attention_type=attention_type,
                sharding=_sharding.MultiHeadSelfAttentionSharding(
                    combined_in_projection=_sharding.LinearSharding(
                        weight=P("X", "Y"), out=P("X", None, "Y")
                    ),
                    out_projection=_sharding.LinearSharding(
                        weight=P("Y", "X"), out=P("X", None, "Y")
                    ),
                ),
            )

            call = (
                nnx.jit(lambda model, x: model(x))
                if use_jit
                else lambda model, x: model(x)
            )

            x = jax.device_put(jnp.array(in_embeddings), P("X", None, "Y"))

            y = call(multi_head_self_attention, x)
            assert y.sharding.spec == P("X", None, "Y")

            multi_head_self_attention.combined_in_projection.weight = jax.device_put(
                jnp.concatenate(
                    [
                        jnp.array(q_proj_weight).transpose(),
                        jnp.array(k_proj_weight).transpose(),
                        jnp.array(v_proj_weight).transpose(),
                    ],
                    axis=-1,
                ),
                P("X", "Y"),
            )
            multi_head_self_attention.out_projection.weight = jax.device_put(
                jnp.array(o_proj_weight).transpose(),
                P("Y", "X"),
            )
            y = call(multi_head_self_attention, x)
            numpy_snapshot.assert_match(y, test_name="test_multihead_self_attention")
            assert y.sharding.spec == P("X", None, "Y")

    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("attention_type", ["custom", "xla"])
    def test_multihead_self_attention_with_rope_sharding(
        self,
        use_jit,
        attention_type,
        numpy_snapshot,
        in_embeddings,
        d_model,
        n_heads,
        ts_state_dict,
        n_keys,
        theta,
        pos_ids,
    ):
        """Test Multi-head self-attention layer with RoPE and FSDP + TP sharding."""
        d, _ = ts_state_dict
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
            d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
        ]
        pos_ids = einops.rearrange(jnp.array(pos_ids), "seq -> 1 seq")
        rope = layers.RoPE(theta=theta, d_k=d_model // n_heads, max_seq_len=n_keys)
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            multi_head_self_attention = layers.MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=n_heads,
                rngs=nnx.Rngs(jax.random.key(42)),
                attention_type=attention_type,
                sharding=_sharding.MultiHeadSelfAttentionSharding(
                    combined_in_projection=_sharding.LinearSharding(
                        weight=P("X", "Y"), out=P("X", None, "Y")
                    ),
                    out_projection=_sharding.LinearSharding(
                        weight=P("Y", "X"), out=P("X", None, "Y")
                    ),
                ),
            )

            call = (
                nnx.jit(
                    lambda model, x, token_positions, rope: model(
                        x, token_positions=token_positions, rope=rope
                    )
                )
                if use_jit
                else lambda model, x, token_positions, rope: model(
                    x, token_positions=token_positions, rope=rope
                )
            )

            x = jax.device_put(jnp.array(in_embeddings), P("X", None, "Y"))
            y = call(
                multi_head_self_attention,
                x,
                token_positions=jnp.array(pos_ids),
                rope=rope,
            )
            assert y.sharding.spec == P("X", None, "Y")

            multi_head_self_attention.combined_in_projection.weight = jax.device_put(
                jnp.concatenate(
                    [
                        jnp.array(q_proj_weight).transpose(),
                        jnp.array(k_proj_weight).transpose(),
                        jnp.array(v_proj_weight).transpose(),
                    ],
                    axis=-1,
                ),
                P("X", "Y"),
            )
            multi_head_self_attention.out_projection.weight = jax.device_put(
                jnp.array(o_proj_weight).transpose(),
                P("Y", "X"),
            )
            y = call(
                multi_head_self_attention,
                x,
                token_positions=jnp.array(pos_ids),
                rope=rope,
            )
            numpy_snapshot.assert_match(
                y, test_name="test_multihead_self_attention_with_rope"
            )
            assert y.sharding.spec == P("X", None, "Y")

    @pytest.mark.parametrize("use_jit", [False, True])
    def test_transformer_block_sharding(
        self,
        use_jit,
        numpy_snapshot,
        ts_state_dict,
        in_embeddings,
        d_model,
        n_heads,
        d_ff,
        n_keys,
        theta,
    ):
        """Test Transformer block with FSDP + TP sharding."""
        block_weights = {
            k.replace("layers.0.", ""): v
            for k, v in ts_state_dict[0].items()
            if "layers.0." in k
        }
        rope = layers.RoPE(theta=theta, d_k=d_model // n_heads, max_seq_len=n_keys)
        mesh = _make_cpu_mesh((4, 2), ("X", "Y"))
        with jax.set_mesh(mesh):
            transformer_block = layers.TransformerBlock(
                d_model=d_model,
                num_heads=n_heads,
                d_ff=d_ff,
                rngs=nnx.Rngs(jax.random.key(42)),
                sharding=_sharding.TransformerBlockSharding(
                    attn=_sharding.MultiHeadSelfAttentionSharding(
                        combined_in_projection=_sharding.LinearSharding(
                            weight=P("X", "Y"), out=P("X", None, "Y")
                        ),
                        out_projection=_sharding.LinearSharding(
                            weight=P("Y", "X"), out=P("X", None, "Y")
                        ),
                    ),
                    ffn=_sharding.SwiGLUSharding(
                        up_projection=_sharding.LinearSharding(
                            weight=P("X", "Y"), out=P("X", None, "Y")
                        ),
                        down_projection=_sharding.LinearSharding(
                            weight=P("Y", "X"), out=P("X", None, "Y")
                        ),
                    ),
                ),
            )

            call = (
                nnx.jit(
                    lambda model, x, rope, token_positions: model(
                        in_features=x,
                        token_positions=token_positions,
                        rope=rope,
                    )
                )
                if use_jit
                else lambda model, x, rope, token_positions: model(
                    in_features=x,
                    token_positions=token_positions,
                    rope=rope,
                )
            )

            x = jax.device_put(jnp.array(in_embeddings), P("X", None, "Y"))
            y = call(
                transformer_block,
                x,
                rope=rope,
                token_positions=jnp.arange(in_embeddings.shape[-2]),
            )
            assert y.sharding.spec == P("X", None, "Y")

            transformer_block.rms_norm_pre_attn.weight = nnx.Param(
                jnp.array(block_weights["ln1.weight"])
            )
            transformer_block.attn.combined_in_projection.weight = nnx.Param(
                jax.device_put(
                    jnp.concatenate(
                        [
                            jnp.array(block_weights["attn.q_proj.weight"]).transpose(),
                            jnp.array(block_weights["attn.k_proj.weight"]).transpose(),
                            jnp.array(block_weights["attn.v_proj.weight"]).transpose(),
                        ],
                        axis=-1,
                    ),
                    P("X", "Y"),
                )
            )
            transformer_block.attn.out_projection.weight = nnx.Param(
                jax.device_put(
                    jnp.array(block_weights["attn.output_proj.weight"]).transpose(),
                    P("Y", "X"),
                )
            )

            transformer_block.rms_norm_pre_ff.weight = nnx.Param(
                jnp.array(block_weights["ln2.weight"])
            )
            transformer_block.ffn.w1_projection.weight = nnx.Param(
                jax.device_put(
                    jnp.array(block_weights["ffn.w1.weight"]).transpose(),
                    P("X", "Y"),
                )
            )
            transformer_block.ffn.w3_projection.weight = nnx.Param(
                jax.device_put(
                    jnp.array(block_weights["ffn.w3.weight"]).transpose(),
                    P("X", "Y"),
                )
            )
            transformer_block.ffn.w2_projection.weight = nnx.Param(
                jax.device_put(
                    jnp.array(block_weights["ffn.w2.weight"]).transpose(),
                    P("Y", "X"),
                )
            )
            y = call(
                transformer_block,
                x,
                rope=rope,
                token_positions=jnp.arange(in_embeddings.shape[-2]),
            )
            numpy_snapshot.assert_match(y, test_name="test_transformer_block")
            assert y.sharding.spec == P("X", None, "Y")


try:
    _cuda_devices = jax.devices("gpu")
    _has_cuda = len(_cuda_devices) > 0
except RuntimeError:
    _has_cuda = False


@pytest.mark.cuda
@pytest.mark.skipif(not _has_cuda, reason="No CUDA device available")
class TestLayersCuda:
    """Tests that require a CUDA device (e.g. cuDNN attention kernels)."""

    @pytest.mark.parametrize("attention_type", ["xla", "cudnn"])
    def test_jax_attention_preserves_tp_sharding(self, attention_type):
        """Verify that the reshard after jax.nn.dot_product_attention restores
        the TP sharding on the model dimension so that out_projection's
        contracting dimensions are consistent.

        Even with a single GPU (mesh of size 1), JAX will raise
        ``ShardingTypeError`` if the contracting-dimension shardings
        disagree, so this reproduces the bug that occurs on multi-GPU
        FSDP+TP setups.
        """
        num_devices = jax.device_count("gpu")
        mesh = jax.make_mesh((num_devices,), ("model",))
        with jax.set_mesh(mesh):
            d_model = 256
            num_heads = 4
            mhsa = layers.MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                rngs=nnx.Rngs(jax.random.key(0)),
                attention_type=attention_type,
                dtype=jnp.bfloat16,
                sharding=_sharding.MultiHeadSelfAttentionSharding(
                    combined_in_projection=_sharding.LinearSharding(
                        weight=P(None, "model"),
                        out=P(None, None, "model"),
                    ),
                    out_projection=_sharding.LinearSharding(
                        weight=P("model", None),
                        out=P(None, None, None),
                    ),
                ),
            )

            @nnx.jit
            def forward(model, x):
                return model(x)

            batch, seq_len = 2, 64
            x = jax.device_put(
                jnp.ones((batch, seq_len, d_model), dtype=jnp.bfloat16),
                P(None, None, "model"),
            )
            # This must not raise ShardingTypeError on the
            # out_projection contracting dimension.
            y = forward(mhsa, x)
            assert y.shape == (batch, seq_len, d_model)

    @pytest.mark.parametrize("attention_type", ["xla", "cudnn"])
    def test_jax_attention_with_rope_preserves_tp_sharding(self, attention_type):
        """Same as above but with RoPE applied."""
        num_devices = jax.device_count("gpu")
        mesh = jax.make_mesh((num_devices,), ("model",))
        with jax.set_mesh(mesh):
            d_model = 256
            num_heads = 4
            d_head = d_model // num_heads
            seq_len = 64
            rope = layers.RoPE(
                theta=10000.0, d_k=d_head, max_seq_len=seq_len, dtype=jnp.bfloat16
            )
            mhsa = layers.MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                rngs=nnx.Rngs(jax.random.key(0)),
                attention_type=attention_type,
                dtype=jnp.bfloat16,
                sharding=_sharding.MultiHeadSelfAttentionSharding(
                    combined_in_projection=_sharding.LinearSharding(
                        weight=P(None, "model"),
                        out=P(None, None, "model"),
                    ),
                    out_projection=_sharding.LinearSharding(
                        weight=P("model", None),
                        out=P(None, None, None),
                    ),
                ),
            )

            @nnx.jit
            def forward(model, x, token_positions, rope):
                return model(x, token_positions=token_positions, rope=rope)

            batch = 2
            x = jax.device_put(
                jnp.ones((batch, seq_len, d_model), dtype=jnp.bfloat16),
                P(None, None, "model"),
            )
            token_positions = jnp.arange(seq_len)[None, :]
            y = forward(mhsa, x, token_positions=token_positions, rope=rope)
            assert y.shape == (batch, seq_len, d_model)

    @pytest.mark.parametrize("attention_type", ["xla", "cudnn"])
    def test_dot_product_attention_tp_sharding(self, attention_type):
        """Document how jax.nn.dot_product_attention handles TP sharding.

        The two implementations behave differently:
        - ``xla``:   preserves the ``"model"`` sharding on the head dimension.
        - ``cudnn``: erases it (returns an unsharded / empty spec).

        In both cases an explicit reshard after the attention call restores the
        expected activation sharding, which is what ``_call_jax_attention``
        does.
        """
        num_devices = jax.device_count("gpu")
        mesh = jax.make_mesh((num_devices,), ("model",))
        with jax.set_mesh(mesh):
            batch, seq_len, num_heads, d_head = 2, 64, 4, 64
            d_model = num_heads * d_head

            # Q/K/V with "model" on the num_heads dimension —
            # mirrors the sharding after combined_in_projection + rearrange.
            q = jax.device_put(
                jnp.ones(
                    (batch, seq_len, num_heads, d_head),
                    dtype=jnp.bfloat16,
                ),
                P(None, None, "model", None),
            )
            k = jax.device_put(jnp.ones_like(q), P(None, None, "model", None))
            v = jax.device_put(jnp.ones_like(q), P(None, None, "model", None))

            # Sanity: inputs carry "model" on axis 2.
            assert q.sharding.spec == P(None, None, "model", None)

            attn_out = jax.nn.dot_product_attention(
                query=q,
                key=k,
                value=v,
                is_causal=True,
                implementation=attention_type,
            )

            spec = attn_out.sharding.spec
            if attention_type == "xla":
                # XLA flash-attention propagates the head-dim sharding.
                assert spec == P(
                    None, None, "model", None
                ), f"xla: unexpected head-dim sharding: {spec}"
            else:
                # cuDNN attention erases the head-dim sharding; the explicit
                # reshard in _call_jax_attention is required to restore it.
                assert spec == P(), (
                    f"cudnn: unexpected head-dim sharding: {spec} _call_jax_attention is safely "
                    "redundant."
                )

            # Rearrange back to (batch, seq_len, d_model), mirroring
            # the rearrange in _call_jax_attention.
            merged = einops.rearrange(
                attn_out,
                "b s h d -> b s (h d)",
            )

            # Reshard restores the expected activation sharding for both
            # implementations.
            activation_sharding = P(None, None, "model")
            resharded = jax.sharding.reshard(merged, activation_sharding)
            assert resharded.sharding.spec == P(None, None, "model")
