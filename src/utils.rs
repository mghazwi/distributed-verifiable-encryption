#![allow(non_snake_case)]
use crate::{
    error::SaverError,
    serde_utils::*,
    concat_slices
};
use ark_ff::{BigInteger, PrimeField};
use ark_std::{fmt::Debug, vec::Vec};
use ark_ec::{scalar_mul::fixed_base::FixedBase, AffineRepr,CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use digest::Digest;

#[cfg(feature = "parallel")]
use {ark_std::cfg_into_iter, rayon::prelude::*};

/// Return number of chunks given the bit size of chunk. Considers the size of the field.
pub fn chunks_count<F: PrimeField>(chunk_bit_size: u8) -> u8 {
    let scalar_size = F::MODULUS_BIT_SIZE as usize;
    let bit_size = chunk_bit_size as usize;
    // ceil(scalar_size / bit_size)
    ((scalar_size + bit_size - 1) / bit_size) as u8
}

pub type CHUNK_TYPE = u16;

/// Given an element `F`, break it into chunks where each chunk is of `chunk_bit_size` bits. This is
/// essentially an n-ary representation where n is `chunk_bit_size`. Returns big-endian representation.
pub fn decompose<F: PrimeField>(message: &F, chunk_bit_size: u8) -> crate::Result<Vec<CHUNK_TYPE>> {
    let bytes = message.into_bigint().to_bytes_be();
    let mut decomposition = Vec::<CHUNK_TYPE>::new();
    match chunk_bit_size {
        4 => {
            for b in bytes {
                decomposition.push((b >> 4) as CHUNK_TYPE);
                decomposition.push((b & 15) as CHUNK_TYPE);
            }
        }
        8 => {
            for b in bytes {
                decomposition.push(b as CHUNK_TYPE);
            }
        }
        16 => {
            // Process 2 bytes at a time
            for bytes_2 in bytes.chunks(2) {
                let mut b = (bytes_2[0] as CHUNK_TYPE) << (8 as CHUNK_TYPE);
                if bytes_2.len() > 1 {
                    b += bytes_2[1] as CHUNK_TYPE;
                }
                decomposition.push(b);
            }
        }
        b => return Err(SaverError::UnexpectedBase(b)),
    }
    Ok(decomposition)
}

/// Recreate a field element back from output of `decompose`. Assumes big-endian representation in `decomposed`
pub fn compose<F: PrimeField>(decomposed: &[CHUNK_TYPE], chunk_bit_size: u8) -> crate::Result<F> {
    match chunk_bit_size {
        4 => {
            if (decomposed.len() % 2) == 1 {
                return Err(SaverError::InvalidDecomposition);
            }
            let mut bytes = Vec::<u8>::with_capacity(decomposed.len() / 2);
            for nibbles in decomposed.chunks(2) {
                bytes.push(((nibbles[0] << 4) + nibbles[1]) as u8);
            }
            Ok(F::from_be_bytes_mod_order(&bytes))
        }
        8 => Ok(F::from_be_bytes_mod_order(
            &decomposed.iter().map(|b| *b as u8).collect::<Vec<u8>>(),
        )),
        16 => {
            let mut bytes = Vec::<u8>::with_capacity(decomposed.len() * 2);
            for byte_2 in decomposed {
                bytes.push((byte_2 >> 8) as u8);
                bytes.push((byte_2 & 255) as u8);
            }
            Ok(F::from_be_bytes_mod_order(&bytes))
        }
        b => Err(SaverError::UnexpectedBase(b)),
    }
}

// msm utils
/// Use when same elliptic curve point is to be multiplied by several scalars.
#[serde_as]
#[derive(
    Clone, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize,
)]
pub struct WindowTable<G: CurveGroup> {
    scalar_size: usize,
    window_size: usize,
    outerc: usize,
    #[serde_as(as = "Vec<Vec<ArkObjectBytes>>")]
    table: Vec<Vec<G::Affine>>,
}

impl<G: CurveGroup> WindowTable<G> {
    /// Create new table for `group_elem`. `num_multiplications` is the number of multiplication that
    /// need to be done and it can be an approximation as it does not impact correctness but only performance.
    pub fn new(num_multiplications: usize, group_elem: G) -> Self {
        let scalar_size = G::ScalarField::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(num_multiplications);
        let outerc = (scalar_size + window_size - 1) / window_size;
        let table = FixedBase::get_window_table(scalar_size, window_size, group_elem);
        Self {
            scalar_size,
            window_size,
            outerc,
            table,
        }
    }

    /// Multiply with a single scalar
    pub fn multiply(&self, element: &G::ScalarField) -> G {
        FixedBase::windowed_mul(self.outerc, self.window_size, &self.table, element)
    }

    /// Multiply with a many scalars
    pub fn multiply_many(&self, elements: &[G::ScalarField]) -> Vec<G> {
        FixedBase::msm(self.scalar_size, self.window_size, &self.table, elements)
    }

    pub fn window_size(num_multiplications: usize) -> usize {
        FixedBase::get_mul_window_size(num_multiplications)
    }
}

/// The same group element is multiplied by each in `elements` using a window table
pub fn multiply_field_elems_with_same_group_elem<G: CurveGroup>(
    group_elem: G,
    elements: &[G::ScalarField],
) -> Vec<G> {
    let table = WindowTable::new(elements.len(), group_elem);
    table.multiply_many(elements)
}

// hashing utils
/// Hash bytes to a point on the curve. Returns as Projective coordinates. This is vulnerable to timing attack and is only used when input
/// is public anyway like when generating setup parameters.
pub fn projective_group_elem_from_try_and_incr<G: AffineRepr, D: Digest>(bytes: &[u8]) -> G::Group {
    let mut hash = D::digest(bytes);
    let mut g = G::from_random_bytes(&hash);
    let mut j = 1u64;
    while g.is_none() {
        hash = D::digest(&concat_slices!(bytes, b"-attempt-", j.to_le_bytes()));
        g = G::from_random_bytes(&hash);
        j += 1;
    }
    g.unwrap().mul_by_cofactor_to_group()
}

/// Hash bytes to a point on the curve. Returns as Affine coordinates. This is vulnerable to timing attack and is only used when input
/// is public anyway like when generating setup parameters.
pub fn affine_group_elem_from_try_and_incr<G: AffineRepr, D: Digest>(bytes: &[u8]) -> G {
    projective_group_elem_from_try_and_incr::<G, D>(bytes).into_affine()
}

/// Hash bytes to a field element. This is vulnerable to timing attack and is only used when input
/// is public anyway like when generating setup parameters or challenge
pub fn field_elem_from_try_and_incr<F: PrimeField, D: Digest>(bytes: &[u8]) -> F {
    let mut hash = D::digest(bytes);
    let mut f = F::from_random_bytes(&hash);
    let mut j = 1u64;
    while f.is_none() {
        hash = D::digest(&concat_slices!(bytes, b"-attempt-", j.to_le_bytes()));
        f = F::from_random_bytes(&hash);
        j += 1;
    }
    f.unwrap()
}


#[cfg(test)]
#[macro_export]
macro_rules! test_serialization {
    ($obj_type:ty, $obj: ident) => {
        let mut serz = vec![];
        CanonicalSerialize::serialize_compressed(&$obj, &mut serz).unwrap();
        let deserz: $obj_type = CanonicalDeserialize::deserialize_compressed(&serz[..]).unwrap();
        assert_eq!(deserz, $obj);

        let mut serz = vec![];
        $obj.serialize_uncompressed(&mut serz).unwrap();
        let deserz: $obj_type = CanonicalDeserialize::deserialize_uncompressed(&serz[..]).unwrap();
        assert_eq!(deserz, $obj);

        // Test JSON serialization
        let ser = serde_json::to_string(&$obj).unwrap();
        let deser = serde_json::from_str::<$obj_type>(&ser).unwrap();
        assert_eq!($obj, deser);

        // Test Message Pack serialization
        let ser = rmp_serde::to_vec_named(&$obj).unwrap();
        let deser = rmp_serde::from_slice::<$obj_type>(&ser).unwrap();
        assert_eq!($obj, deser);
    };
}

